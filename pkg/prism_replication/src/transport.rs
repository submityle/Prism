//! The transport seam and a deterministic in-memory link (design §6).
//!
//! Real multiplayer rides a socket (reliable UDP / QUIC), which this crate does
//! not depend on. What it *can* define without a network — and test on one
//! machine — is the shape of a transport and a **loopback link** with a
//! deterministic fault model (latency, jitter, loss). That link is a real
//! bidirectional byte channel usable for local play and tests, and it lets the
//! [`crate::sequence`] delivery window be exercised against reordering and loss
//! exactly as it would be over UDP.
//!
//! A message is opaque bytes: the sequencing / delta layers above decide their
//! own framing. Delivery within a [`poll`](LoopbackLink::poll) is ordered by
//! `(deliver_at, send order)`, so jitter deliberately reorders messages relative
//! to send order — and a given configuration always reorders and drops the
//! *same* messages, so runs are reproducible.

use alloc::boxed::Box;
use alloc::vec::Vec;

/// The transport provider seam (design §6): bytes out, bytes in.
///
/// A real provider (reliable UDP / QUIC) implements this over a socket; the
/// in-memory pair below implements it over a shared queue, so an app can be wired
/// end-to-end without a network. `Transport` is `no_std`-clean — the std-only
/// pieces are the concrete in-memory endpoints.
pub trait Transport: Send + Sync {
    /// Send `bytes` to the peer.
    fn send(&mut self, bytes: &[u8]);
    /// Take the next inbound message, if any.
    fn recv(&mut self) -> Option<Vec<u8>>;
}

/// A boxed transport is itself a transport, so a caller can erase the concrete
/// provider (a [`reliable::ReliableChannel`](crate::reliable::ReliableChannel)
/// over a `Box<dyn Transport>` still works).
impl<T: Transport + ?Sized> Transport for Box<T> {
    fn send(&mut self, bytes: &[u8]) {
        (**self).send(bytes);
    }
    fn recv(&mut self) -> Option<Vec<u8>> {
        (**self).recv()
    }
}

/// A shared in-memory [`Transport`] pair (std): one endpoint's `send` lands in the
/// other's inbox immediately, so an app can be driven end-to-end in tests and
/// local play without a socket.
#[cfg(feature = "std")]
pub mod memory {
    use super::Transport;
    use alloc::collections::VecDeque;
    use alloc::sync::Arc;
    use alloc::vec::Vec;
    use std::sync::Mutex;

    #[derive(Default)]
    struct Inboxes {
        to_a: VecDeque<Vec<u8>>,
        to_b: VecDeque<Vec<u8>>,
    }

    /// One end of a shared in-memory transport pair.
    #[derive(Clone)]
    pub struct InMemoryTransport {
        inboxes: Arc<Mutex<Inboxes>>,
        as_a: bool,
    }

    impl InMemoryTransport {
        /// Two connected endpoints `(a, b)`.
        #[must_use]
        pub fn pair() -> (Self, Self) {
            let inboxes = Arc::new(Mutex::new(Inboxes::default()));
            (
                Self {
                    inboxes: Arc::clone(&inboxes),
                    as_a: true,
                },
                Self {
                    inboxes,
                    as_a: false,
                },
            )
        }

        /// How many messages are waiting to be received by this endpoint.
        #[must_use]
        pub fn pending(&self) -> usize {
            let inboxes = lock(&self.inboxes);
            if self.as_a {
                inboxes.to_a.len()
            } else {
                inboxes.to_b.len()
            }
        }
    }

    impl Transport for InMemoryTransport {
        fn send(&mut self, bytes: &[u8]) {
            let mut inboxes = lock(&self.inboxes);
            let target = if self.as_a {
                &mut inboxes.to_b
            } else {
                &mut inboxes.to_a
            };
            target.push_back(bytes.to_vec());
        }

        fn recv(&mut self) -> Option<Vec<u8>> {
            let mut inboxes = lock(&self.inboxes);
            let source = if self.as_a {
                &mut inboxes.to_a
            } else {
                &mut inboxes.to_b
            };
            source.pop_front()
        }
    }

    fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
        mutex
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn the_pair_delivers_each_way_once() {
            let (mut a, mut b) = InMemoryTransport::pair();
            a.send(b"from-a"); // lands in B's inbox
            b.send(b"from-b"); // lands in A's inbox
            assert_eq!(b.pending(), 1, "B has A's message");
            assert_eq!(a.pending(), 1, "A has B's message");
            assert_eq!(a.recv(), Some(b"from-b".to_vec()));
            assert_eq!(b.recv(), Some(b"from-a".to_vec()));
            assert_eq!(a.recv(), None);
            assert_eq!(b.pending(), 0);
        }
    }
}

/// A deterministic two-endpoint link with latency, jitter and loss.
///
/// Delivery is driven by [`poll`](Self::poll): each call advances one logical
/// tick and returns the messages that have arrived for the endpoint being
/// polled, oldest first. Jitter comes from a seeded `SplitMix64`, so a given
/// configuration always drops and reorders the same messages.
#[derive(Clone, Debug)]
pub struct LoopbackLink {
    tick: u64,
    latency: u32,
    jitter: u32,
    drop_every: Option<u64>,
    sent: u64,
    dropped: u64,
    in_flight: Vec<InFlight>,
    rng: SplitMix64,
}

#[derive(Clone, Debug)]
struct InFlight {
    deliver_at: u64,
    /// Monotone send order, the tie-break so deliveries are deterministic.
    seq: u64,
    to_a: bool,
    bytes: Vec<u8>,
}

impl LoopbackLink {
    /// A link with a base latency of `latency` ticks and up to `jitter` extra
    /// ticks of per-message variation (both in logical ticks).
    ///
    /// The delay parameters are `u32` ticks deliberately: it keeps
    /// `tick + latency + jitter` far from `u64` saturation, so the drain idiom
    /// `while link.in_flight() > 0 { poll(..) }` always terminates (a `u64`
    /// latency near `u64::MAX` would pin messages at an unreachable tick).
    #[must_use]
    pub fn new(latency: u32, jitter: u32) -> Self {
        Self {
            tick: 0,
            latency,
            jitter,
            drop_every: None,
            sent: 0,
            dropped: 0,
            in_flight: Vec::new(),
            rng: SplitMix64::new(0x5EED_1234_ABCD_0001),
        }
    }

    /// Drop every `n`th send, deterministically (`n == 0` disables loss). Builder
    /// style.
    #[must_use]
    pub fn with_drop_every(mut self, n: u64) -> Self {
        self.drop_every = (n > 0).then_some(n);
        self
    }

    /// The current logical tick.
    #[must_use]
    pub fn tick(&self) -> u64 {
        self.tick
    }

    /// How many messages are still in flight.
    #[must_use]
    pub fn in_flight(&self) -> usize {
        self.in_flight.len()
    }

    /// How many sends were dropped by the fault model.
    #[must_use]
    pub fn dropped(&self) -> u64 {
        self.dropped
    }

    /// Send `bytes` from endpoint A (`from_a == true`) or B, returning whether it
    /// was accepted (a dropped message returns `false`).
    pub fn send(&mut self, from_a: bool, bytes: Vec<u8>) -> bool {
        self.sent = self.sent.saturating_add(1);
        if let Some(n) = self.drop_every
            && self.sent.is_multiple_of(n)
        {
            self.dropped += 1;
            return false;
        }
        let extra = if self.jitter > 0 {
            self.rng.next_below(u64::from(self.jitter) + 1)
        } else {
            0
        };
        self.in_flight.push(InFlight {
            deliver_at: self
                .tick
                .saturating_add(u64::from(self.latency))
                .saturating_add(extra),
            seq: self.sent,
            to_a: !from_a,
            bytes,
        });
        true
    }

    /// Advance one tick and return the messages that have arrived at endpoint A
    /// (`at_a == true`) or B, oldest first.
    pub fn poll(&mut self, at_a: bool) -> Vec<Vec<u8>> {
        self.tick = self.tick.saturating_add(1);
        // Select the due messages, then remove them. `retain` keeps the remaining
        // in-flight messages in send order.
        let mut due: Vec<InFlight> = Vec::new();
        let mut kept: Vec<InFlight> = Vec::with_capacity(self.in_flight.len());
        for message in self.in_flight.drain(..) {
            if message.to_a == at_a && message.deliver_at <= self.tick {
                due.push(message);
            } else {
                kept.push(message);
            }
        }
        self.in_flight = kept;
        due.sort_by_key(|message| (message.deliver_at, message.seq));
        due.into_iter().map(|message| message.bytes).collect()
    }
}

/// A tiny `SplitMix64` PRNG, used only to make fault injection reproducible.
#[derive(Clone, Debug)]
struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A value in `0..bound` (`0` when `bound` is zero).
    fn next_below(&mut self, bound: u64) -> u64 {
        if bound == 0 { 0 } else { self.next_u64() % bound }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sequence::{Delivery, DeliveryWindow};
    use alloc::vec;

    #[test]
    fn a_clean_link_delivers_every_message() {
        let mut link = LoopbackLink::new(2, 0);
        assert!(link.send(true, vec![1]));
        assert!(link.send(true, vec![2]));
        assert_eq!(link.in_flight(), 2);

        // Nothing arrives before the latency elapses.
        assert!(link.poll(false).is_empty());
        assert_eq!(link.poll(false), vec![vec![1], vec![2]]);
        assert_eq!(link.in_flight(), 0);
    }

    #[test]
    fn the_drop_model_is_deterministic() {
        let mut link = LoopbackLink::new(1, 0).with_drop_every(5);
        let accepted: Vec<bool> = (0..20).map(|i| link.send(true, vec![i as u8])).collect();
        assert_eq!(link.dropped(), 4, "every 5th of 20 sends is dropped");
        assert!(!accepted[4] && !accepted[9]);

        // A second, identically configured link drops the same sends.
        let mut other = LoopbackLink::new(1, 0).with_drop_every(5);
        let other_accepted: Vec<bool> = (0..20).map(|i| other.send(true, vec![i as u8])).collect();
        assert_eq!(accepted, other_accepted);
    }

    /// The delay parameters are `u32`, so `latency + jitter` stays far below the
    /// `u64` tick ceiling — the drain idiom always terminates and jitter cannot
    /// overflow (regression: `jitter + 1` overflowed at `u64::MAX`, and a near-
    /// `u64::MAX` latency pinned a message at an unreachable tick).
    #[test]
    fn extreme_fault_parameters_do_not_overflow() {
        let mut link = LoopbackLink::new(u32::MAX, u32::MAX);
        assert!(link.send(true, vec![7]));
        assert_eq!(
            link.in_flight(),
            1,
            "building the delivery tick did not overflow"
        );
    }

    /// The delivery window, fed over the lossy/jittery link, restores in-order
    /// delivery up to the first dropped sequence — exactly its contract.
    #[test]
    fn the_delivery_window_restores_order_over_a_lossy_link() {
        let mut link = LoopbackLink::new(1, 3).with_drop_every(5);
        let mut window: DeliveryWindow<u64> = DeliveryWindow::new(16);

        for seq in 0..20u64 {
            link.send(true, seq.to_le_bytes().to_vec());
        }

        let mut delivered = 0u64;
        while link.in_flight() > 0 {
            for bytes in link.poll(false) {
                let seq = u64::from_le_bytes(bytes.try_into().expect("eight bytes"));
                if let Delivery::Advanced(run) = window.insert(seq, seq) {
                    for (_, value) in run {
                        assert_eq!(value, delivered, "delivered strictly in order");
                        delivered += 1;
                    }
                }
            }
        }
        // Sequence 4 was dropped (the 5th send), so frames 0..4 (four of them)
        // are delivered and everything after waits behind the gap.
        assert_eq!(delivered, 4);
        assert_eq!(window.next(), 4);
        assert!(window.buffered() > 0, "later frames wait behind the gap");
    }
}
