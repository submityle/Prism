//! The reliable-ordered channel: ARQ wired onto a datagram transport (design §6).
//!
//! [`crate::arq`] is the protocol (sequence, ACK, retransmit, adaptive RTO) and
//! [`crate::transport`] is the best-effort byte seam — but neither knows the
//! other, and the session above needs **one** object that speaks reliable ordered
//! messages. [`ReliableChannel`] is that glue and nothing more: it owns one
//! [`ReliableSender`] and one [`ReliableReceiver`], encodes each message as a
//! small datagram (`[kind: u8][seq: u64 LE][payload..]`), drives both halves on
//! [`tick`](ReliableChannel::tick) to retransmit and drain, and hands the
//! reassembled, in-order payloads up to the caller. It adds no protocol of its
//! own, so the whole thing is deterministic and testable without a socket.
//!
//! Both peers of a session run a `ReliableChannel`: the sender numbers each
//! payload and keeps it until an ACK arrives, and the receiver reassembles in
//! order and ACKs what it has. Drop it over the `std` `udp::UdpTransport` for a
//! real network, or over an in-memory [`Transport`] for tests and local play.

use alloc::collections::VecDeque;
use alloc::vec::Vec;

use crate::arq::{ReliableReceiver, ReliableSender};
use crate::transport::Transport;

/// The datagram kind byte for a data packet.
const KIND_DATA: u8 = 0;
/// The datagram kind byte for an ACK.
const KIND_ACK: u8 = 1;
/// Every packet starts with a kind byte and a `u64` sequence number.
const HEADER_LEN: usize = 9;

/// Encode a data packet: `[0][seq: u64 LE][payload..]`. `pub(crate)` so the
/// protocol tests exercise the exact production codec.
pub(crate) fn data_packet(seq: u64, payload: &[u8]) -> Vec<u8> {
    let mut packet = Vec::with_capacity(HEADER_LEN + payload.len());
    packet.push(KIND_DATA);
    packet.extend_from_slice(&seq.to_le_bytes());
    packet.extend_from_slice(payload);
    packet
}

/// Encode an ACK packet: `[1][seq: u64 LE]`.
pub(crate) fn ack_packet(seq: u64) -> Vec<u8> {
    let mut packet = Vec::with_capacity(HEADER_LEN);
    packet.push(KIND_ACK);
    packet.extend_from_slice(&seq.to_le_bytes());
    packet
}

/// Decode a packet header, or `None` when it is too short to be one (a truncated
/// or hostile datagram is ignored, never a panic).
pub(crate) fn decode(packet: &[u8]) -> Option<(u8, u64, &[u8])> {
    let header = packet.get(..HEADER_LEN)?;
    let seq = u64::from_le_bytes(header[1..].try_into().expect("eight bytes"));
    Some((header[0], seq, &packet[HEADER_LEN..]))
}

/// Reliable ordered message delivery over a best-effort datagram [`Transport`].
///
/// `tick` is the clock: it is driven once per frame (or once per fixed step), and
/// retransmit timeouts are measured in those ticks. The caller drives it and then
/// reads reassembled messages with [`recv`](Self::recv).
pub struct ReliableChannel<D: Transport> {
    datagram: D,
    sender: ReliableSender,
    receiver: ReliableReceiver,
    now: u64,
    inbound: VecDeque<Vec<u8>>,
}

impl<D: Transport> ReliableChannel<D> {
    /// Wrap `datagram` in a reliable ordered channel.
    ///
    /// `min_rto` is the retransmit timeout floor in ticks; `window` bounds how many
    /// messages may be unacknowledged at once (the sender refuses beyond it, see
    /// [`ReliableSender::send`]) and how many out-of-order messages the receiver
    /// buffers before it refuses (see [`ReliableReceiver::new`]).
    #[must_use]
    pub fn new(datagram: D, min_rto: u32, window: usize) -> Self {
        Self {
            datagram,
            sender: ReliableSender::new(min_rto, window),
            receiver: ReliableReceiver::new(window),
            now: 0,
            inbound: VecDeque::new(),
        }
    }

    /// Queue `payload` for reliable ordered delivery, returning its sequence — or
    /// `None` when the send window is full (retry after a later tick when
    /// acknowledgements have returned).
    pub fn send(&mut self, payload: &[u8]) -> Option<u64> {
        self.sender.send(payload.to_vec())
    }

    /// Advance one tick and move bytes both ways: drain inbound datagrams (and
    /// their ACKs), transmit new / timed-out messages, and reassemble what is now
    /// deliverable.
    pub fn tick(&mut self) {
        let now = self.now.saturating_add(1);
        self.tick_at(now);
    }

    /// Advance to an explicit `now` and move bytes both ways (for a caller whose
    /// clock is not a simple per-frame counter, and for deterministic tests).
    pub fn tick_at(&mut self, now: u64) {
        // The clock must be monotonic; clamp so a jittery or regressing source
        // (or a `now` at the very top of the range) cannot move time backwards and
        // corrupt the RTO math.
        self.now = now.max(self.now);
        self.pump();
    }

    /// Take the next recovered message, in order, if one is ready.
    pub fn recv(&mut self) -> Option<Vec<u8>> {
        self.inbound.pop_front()
    }

    /// The current logical tick (monotonic; a regressing [`tick_at`](Self::tick_at)
    /// is clamped forward).
    #[must_use]
    pub fn now(&self) -> u64 {
        self.now
    }

    /// How many messages are still unacknowledged.
    #[must_use]
    pub fn unacked(&self) -> usize {
        self.sender.unacked()
    }

    /// Whether the send window is full (a [`send`](Self::send) would return
    /// `None`); the caller should wait for acknowledgements before retrying.
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.sender.is_full()
    }

    /// The current retransmit timeout, in ticks.
    #[must_use]
    pub fn rto(&self) -> u32 {
        self.sender.rto()
    }

    /// The smoothed round-trip estimate, once an ACK has been sampled.
    #[must_use]
    pub fn srtt(&self) -> Option<u32> {
        self.sender.srtt()
    }

    /// The wrapped datagram transport, for a caller that needs to inspect or
    /// reconfigure it (e.g. a test injecting loss).
    pub fn datagram_mut(&mut self) -> &mut D {
        &mut self.datagram
    }

    /// One exchange step: consume inbound datagrams (and their ACKs), then
    /// retransmit what is due and send new messages.
    fn pump(&mut self) {
        let now = self.now;
        // Drain inbound first, so an ACK that is already waiting retires its
        // message before the transmit phase decides what to retransmit — this
        // avoids a same-tick, off-by-one spurious resend.
        while let Some(packet) = self.datagram.recv() {
            let Some((kind, seq, payload)) = decode(&packet) else {
                continue;
            };
            match kind {
                KIND_DATA => {
                    let (delivered, ack) = self.receiver.receive(seq, payload.to_vec());
                    self.inbound.extend(delivered);
                    if let Some(ack) = ack {
                        self.datagram.send(&ack_packet(ack));
                    }
                }
                KIND_ACK => {
                    self.sender.ack(seq, now);
                }
                _ => {}
            }
        }
        // Then transmit new messages and any retransmissions that are now due.
        let mut outgoing = Vec::new();
        self.sender.tick(now, &mut outgoing);
        for (seq, payload) in outgoing {
            self.datagram.send(&data_packet(seq, &payload));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    /// A capture transport: `send` records into `outbox`, `recv` draws from
    /// `inbox`. The test moves bytes between two of them by hand, so loss,
    /// reordering and timing are all explicit and deterministic — no socket.
    #[derive(Default)]
    struct MockTransport {
        outbox: Vec<Vec<u8>>,
        inbox: VecDeque<Vec<u8>>,
    }

    impl Transport for MockTransport {
        fn send(&mut self, bytes: &[u8]) {
            self.outbox.push(bytes.to_vec());
        }
        fn recv(&mut self) -> Option<Vec<u8>> {
            self.inbox.pop_front()
        }
    }

    /// Hand every captured datagram from `from` to `to` (each direction is moved
    /// by its own call, so a test can drop selectively instead).
    fn hand(from: &mut MockTransport, to: &mut MockTransport) {
        to.inbox.extend(from.outbox.drain(..));
    }

    #[test]
    fn messages_are_delivered_in_order_and_acknowledged() {
        let mut a: ReliableChannel<MockTransport> =
            ReliableChannel::new(MockTransport::default(), 2, 8);
        let mut b: ReliableChannel<MockTransport> =
            ReliableChannel::new(MockTransport::default(), 2, 8);
        assert!(a.send(b"one").is_some());
        assert!(a.send(b"two").is_some());

        let mut got = Vec::new();
        for _ in 0..10 {
            a.tick();
            hand(a.datagram_mut(), b.datagram_mut());
            b.tick();
            hand(b.datagram_mut(), a.datagram_mut());
            a.tick();
            hand(a.datagram_mut(), b.datagram_mut());
            while let Some(message) = b.recv() {
                got.push(message);
            }
            if got.len() == 2 && a.unacked() == 0 {
                break;
            }
        }

        assert_eq!(got, vec![b"one".to_vec(), b"two".to_vec()]);
        assert_eq!(a.unacked(), 0, "both messages acknowledged");
        assert_eq!(b.unacked(), 0);
        assert!(a.srtt().is_some(), "the ACK taught the sender its RTT");
    }

    /// A lost datagram must be retransmitted (not dropped): the receiver still
    /// gets exactly one copy, in order, and the sender eventually forgets it.
    #[test]
    fn a_lost_datagram_is_retransmitted_and_not_duplicated() {
        let mut a: ReliableChannel<MockTransport> =
            ReliableChannel::new(MockTransport::default(), 2, 8);
        let mut b: ReliableChannel<MockTransport> =
            ReliableChannel::new(MockTransport::default(), 2, 8);
        assert!(a.send(b"payload").is_some());

        // The first transmission is dropped on the floor.
        a.tick();
        a.datagram_mut().outbox.clear();

        let mut got = Vec::new();
        for _ in 0..20 {
            a.tick();
            hand(a.datagram_mut(), b.datagram_mut());
            b.tick();
            hand(b.datagram_mut(), a.datagram_mut());
            a.tick();
            hand(a.datagram_mut(), b.datagram_mut());
            while let Some(message) = b.recv() {
                got.push(message);
            }
            if got.len() == 1 && a.unacked() == 0 {
                break;
            }
        }

        assert_eq!(got, vec![b"payload".to_vec()], "recovered exactly once");
        assert_eq!(a.unacked(), 0, "the retransmitted message was acknowledged");
    }

    /// A duplicate delivery (e.g. an ACK was lost so the sender resent) is not
    /// handed up twice.
    #[test]
    fn a_retransmit_after_a_lost_ack_is_not_delivered_twice() {
        let mut a: ReliableChannel<MockTransport> =
            ReliableChannel::new(MockTransport::default(), 2, 8);
        let mut b: ReliableChannel<MockTransport> =
            ReliableChannel::new(MockTransport::default(), 2, 8);
        assert!(a.send(b"once").is_some());

        let mut got = Vec::new();
        for step in 0..20 {
            a.tick();
            hand(a.datagram_mut(), b.datagram_mut());
            b.tick();
            // Drop B's ACK the first time, forcing A to retransmit.
            if step == 0 {
                b.datagram_mut().outbox.clear();
            }
            hand(b.datagram_mut(), a.datagram_mut());
            while let Some(message) = b.recv() {
                got.push(message);
            }
            if got.len() == 1 && a.unacked() == 0 {
                break;
            }
        }

        assert_eq!(got, vec![b"once".to_vec()], "delivered exactly once");
        assert_eq!(a.unacked(), 0);
    }

    /// A truncated / hostile datagram is ignored, never a panic.
    #[test]
    fn a_truncated_datagram_is_ignored() {
        assert!(decode(&[]).is_none());
        assert!(
            decode(&[KIND_DATA, 0, 0, 0]).is_none(),
            "four bytes is short"
        );
        assert!(
            decode(&[KIND_DATA, 1, 0, 0, 0, 0, 0, 0, 0]).is_some(),
            "exactly a header"
        );

        let mut a: ReliableChannel<MockTransport> =
            ReliableChannel::new(MockTransport::default(), 2, 4);
        a.datagram_mut().inbox.push_back(vec![0xFF; 2]);
        a.tick();
        assert!(a.recv().is_none(), "the junk produced nothing");
    }

    /// A regressing clock is clamped forward, so a jittery external source cannot
    /// move time backwards and corrupt the RTO math.
    #[test]
    fn the_clock_never_moves_backwards() {
        let mut a: ReliableChannel<MockTransport> =
            ReliableChannel::new(MockTransport::default(), 2, 8);
        a.tick_at(10);
        assert_eq!(a.now(), 10);
        a.tick_at(3);
        assert_eq!(a.now(), 10, "a regressing clock is clamped");
        a.tick();
        assert_eq!(a.now(), 11, "the per-frame tick still advances");
    }
}

/// The real end-to-end wiring: two [`ReliableChannel`]s over real loopback UDP.
/// Gated on `std`; `PRISM_REQUIRE_NET=1` makes a bind failure a hard error rather
/// than a skip, so a CI run can prove the socket path really executed.
#[cfg(all(test, feature = "std"))]
mod udp_tests {
    use super::*;
    use crate::udp::UdpTransport;
    use alloc::vec;
    use std::time::Duration;

    fn require_net() -> bool {
        std::env::var_os("PRISM_REQUIRE_NET").is_some()
    }

    /// Bind a loopback transport, or skip when the sandbox forbids sockets.
    fn bind_or_skip() -> Option<UdpTransport> {
        match UdpTransport::bind("127.0.0.1:0") {
            Ok(transport) => Some(transport),
            Err(error) => {
                assert!(
                    !require_net(),
                    "PRISM_REQUIRE_NET is set but binding a UDP socket failed: {error}"
                );
                None
            }
        }
    }

    #[test]
    fn arq_delivers_every_message_over_real_udp() {
        let Some(mut a_socket) = bind_or_skip() else {
            return;
        };
        let Some(mut b_socket) = bind_or_skip() else {
            return;
        };
        let a_addr = a_socket.local_addr().expect("a local addr");
        let b_addr = b_socket.local_addr().expect("b local addr");
        a_socket.connect_to(b_addr).expect("a peer");
        b_socket.connect_to(a_addr).expect("b peer");

        let mut a: ReliableChannel<UdpTransport> = ReliableChannel::new(a_socket, 2, 8);
        let mut b: ReliableChannel<UdpTransport> = ReliableChannel::new(b_socket, 2, 8);
        for i in 0..4u8 {
            assert!(a.send(&[i]).is_some());
        }

        let mut got = Vec::new();
        for _ in 0..2_000 {
            a.tick();
            b.tick();
            a.tick();
            while let Some(message) = b.recv() {
                got.push(message);
            }
            if got.len() == 4 && a.unacked() == 0 {
                break;
            }
            std::thread::sleep(Duration::from_millis(1));
        }

        assert_eq!(
            got,
            (0..4u8).map(|i| vec![i]).collect::<Vec<_>>(),
            "all four messages, in order, exactly once"
        );
        assert_eq!(a.unacked(), 0, "every datagram was acknowledged over UDP");
    }
}
