//! The pluggable remote-tool channel abstraction and an in-process channel.
//!
//! A transport is a single bidirectional endpoint: the holder pushes
//! `Outbound` messages toward its peer and polls for `Inbound` messages the
//! peer sent back. From a tool the outbound direction carries
//! [`crate::command::AuthoringCommand`] values and the inbound direction
//! carries telemetry; the engine endpoint sees the two directions reversed.
//! The trait is transport-neutral so a real deployment can back it with a local
//! socket, while [`InProcessTransport`] provides a dependency-free,
//! `no_std`-compatible pair for same-process tools and tests.
//!
//! The in-process pair is built from two `alloc` queues shared through
//! reference counting; it never uses `std`-only primitives such as channels, so
//! the core data structure compiles on `no_std` targets.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio,
//! Dolby, or Google Resonance Audio source or derived code; no AI/ML.
//!
//! # Relationship
//! Implements the remote-tool channel of design section 38. The abstraction is
//! intentionally direction-generic so command traffic (toward design section
//! 21's command ring) and telemetry traffic (from design section 26's ring)
//! ride the same endpoint type with the roles swapped at each end.

use alloc::collections::VecDeque;
use alloc::rc::Rc;
use core::cell::RefCell;

/// Default per-direction queue capacity used by [`InProcessTransport::pair`].
pub const DEFAULT_CAPACITY: usize = 256;

/// Why a transport operation could not complete.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(serde::Serialize, serde::Deserialize))]
pub enum TransportError {
    /// The peer endpoint has been dropped, so the link is gone.
    Disconnected,
    /// The outbound queue is at capacity; the caller should retry later.
    Full,
}

/// A single bidirectional endpoint of a remote authoring link.
///
/// Implementors move `Outbound` messages toward the peer and surface `Inbound`
/// messages from it. Both operations are non-blocking: [`AuthoringTransport::poll`]
/// returns [`None`] when nothing is waiting rather than parking the caller,
/// which keeps the trait usable from a cooperative tool loop.
pub trait AuthoringTransport {
    /// The message type this endpoint sends toward its peer.
    type Outbound;
    /// The message type this endpoint receives from its peer.
    type Inbound;

    /// Enqueues `message` for delivery to the peer.
    ///
    /// # Errors
    ///
    /// Returns [`TransportError::Disconnected`] if the peer is gone, or
    /// [`TransportError::Full`] if the outbound queue is saturated.
    fn send(&mut self, message: Self::Outbound) -> Result<(), TransportError>;

    /// Removes and returns the next pending inbound message, if any.
    fn poll(&mut self) -> Option<Self::Inbound>;

    /// Returns `true` while the peer endpoint is still alive.
    fn is_connected(&self) -> bool;
}

/// A bounded, single-process, bidirectional transport backed by two shared
/// `alloc` queues.
///
/// Construct a connected pair with [`InProcessTransport::pair`]; dropping one
/// half marks the other [`disconnected`](AuthoringTransport::is_connected).
/// The type is single-threaded (it uses [`Rc`]), which is exactly the model for
/// an editor talking to an engine inside one process or for a unit test.
pub struct InProcessTransport<O, I> {
    outbound: Rc<RefCell<VecDeque<O>>>,
    inbound: Rc<RefCell<VecDeque<I>>>,
    capacity: usize,
}

impl<O, I> InProcessTransport<O, I> {
    /// Creates a connected pair of endpoints with [`DEFAULT_CAPACITY`].
    ///
    /// The first endpoint sends `A` and receives `B`; the second sends `B` and
    /// receives `A`.
    #[must_use]
    pub fn pair() -> (InProcessTransport<O, I>, InProcessTransport<I, O>) {
        Self::pair_with_capacity(DEFAULT_CAPACITY)
    }

    /// Creates a connected pair of endpoints with a per-direction `capacity`.
    ///
    /// A `capacity` of zero is treated as one so at least a single message can
    /// always be in flight.
    #[must_use]
    pub fn pair_with_capacity(
        capacity: usize,
    ) -> (InProcessTransport<O, I>, InProcessTransport<I, O>) {
        let cap = capacity.max(1);
        let o_to_i: Rc<RefCell<VecDeque<O>>> = Rc::new(RefCell::new(VecDeque::new()));
        let i_to_o: Rc<RefCell<VecDeque<I>>> = Rc::new(RefCell::new(VecDeque::new()));
        let first = InProcessTransport {
            outbound: Rc::clone(&o_to_i),
            inbound: Rc::clone(&i_to_o),
            capacity: cap,
        };
        let second = InProcessTransport {
            outbound: i_to_o,
            inbound: o_to_i,
            capacity: cap,
        };
        (first, second)
    }

    /// Returns the per-direction queue capacity of this endpoint.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns the number of inbound messages currently waiting to be polled.
    #[must_use]
    pub fn pending(&self) -> usize {
        self.inbound.borrow().len()
    }
}

impl<O, I> AuthoringTransport for InProcessTransport<O, I> {
    type Outbound = O;
    type Inbound = I;

    fn send(&mut self, message: O) -> Result<(), TransportError> {
        // A live peer holds the second reference to our outbound queue.
        if Rc::strong_count(&self.outbound) < 2 {
            return Err(TransportError::Disconnected);
        }
        let mut queue = self.outbound.borrow_mut();
        if queue.len() >= self.capacity {
            return Err(TransportError::Full);
        }
        queue.push_back(message);
        Ok(())
    }

    fn poll(&mut self) -> Option<I> {
        self.inbound.borrow_mut().pop_front()
    }

    fn is_connected(&self) -> bool {
        // Either shared queue retaining a second owner means the peer lives.
        Rc::strong_count(&self.outbound) >= 2 || Rc::strong_count(&self.inbound) >= 2
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::command::{AuthoringCommand, EventId};

    #[test]
    fn command_flows_one_way() {
        let (mut tool, mut engine) =
            InProcessTransport::<AuthoringCommand, u32>::pair();
        let cmd = AuthoringCommand::TriggerEvent {
            event: EventId(42),
        };
        assert_eq!(tool.send(cmd), Ok(()));
        assert_eq!(engine.poll(), Some(cmd));
        assert_eq!(engine.poll(), None);
    }

    #[test]
    fn telemetry_flows_the_other_way() {
        let (mut tool, mut engine) =
            InProcessTransport::<AuthoringCommand, u32>::pair();
        assert_eq!(engine.send(99), Ok(()));
        assert_eq!(tool.poll(), Some(99));
    }

    #[test]
    fn capacity_is_enforced() {
        let (mut tool, _engine) =
            InProcessTransport::<u8, u8>::pair_with_capacity(2);
        assert_eq!(tool.send(1), Ok(()));
        assert_eq!(tool.send(2), Ok(()));
        assert_eq!(tool.send(3), Err(TransportError::Full));
    }

    #[test]
    fn dropping_peer_disconnects() {
        let (mut tool, engine) = InProcessTransport::<u8, u8>::pair();
        assert!(tool.is_connected());
        drop(engine);
        assert!(!tool.is_connected());
        assert_eq!(tool.send(1), Err(TransportError::Disconnected));
    }

    #[test]
    fn pending_counts_inbound() {
        let (tool, mut engine) = InProcessTransport::<u8, u8>::pair();
        assert_eq!(tool.pending(), 0);
        engine.send(7).unwrap();
        engine.send(8).unwrap();
        assert_eq!(tool.pending(), 2);
    }
}
