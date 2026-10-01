//! Live regions and announcement queues.
//!
//! A [`LiveRegion`] models an area whose updates are announced by a screen
//! reader. Announcements are buffered in FIFO order and flushed with
//! [`drain`](LiveRegion::drain); the [`Politeness`] setting tells the consumer
//! whether to interrupt the user.

use alloc::string::String;
use alloc::vec::Vec;

/// How insistently a live region's updates should be announced.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Politeness {
    /// Announce at the next graceful opportunity, without interrupting.
    Polite,
    /// Announce immediately, interrupting current speech.
    Assertive,
}

impl Politeness {
    /// Returns `true` when updates interrupt current speech.
    #[must_use]
    pub fn interrupts(&self) -> bool {
        matches!(self, Politeness::Assertive)
    }
}

/// A buffered live region.
#[derive(Clone, Debug)]
pub struct LiveRegion {
    politeness: Politeness,
    queue: Vec<String>,
}

impl LiveRegion {
    /// Creates a live region with the given [`Politeness`].
    #[must_use]
    pub fn new(politeness: Politeness) -> Self {
        Self {
            politeness,
            queue: Vec::new(),
        }
    }

    /// Creates a `polite` live region.
    #[must_use]
    pub fn polite() -> Self {
        Self::new(Politeness::Polite)
    }

    /// Creates an `assertive` live region.
    #[must_use]
    pub fn assertive() -> Self {
        Self::new(Politeness::Assertive)
    }

    /// The region's politeness.
    #[must_use]
    pub fn politeness(&self) -> Politeness {
        self.politeness
    }

    /// Enqueues a message to be announced, preserving order.
    pub fn announce(&mut self, message: impl Into<String>) {
        self.queue.push(message.into());
    }

    /// The number of pending announcements.
    #[must_use]
    pub fn pending(&self) -> usize {
        self.queue.len()
    }

    /// Returns `true` when there are no pending announcements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    /// Removes and returns all pending announcements in FIFO order, leaving the
    /// queue empty.
    pub fn drain(&mut self) -> Vec<String> {
        core::mem::take(&mut self.queue)
    }
}

#[cfg(test)]
mod tests {
    use super::{LiveRegion, Politeness};
    use alloc::string::String;
    use alloc::vec::Vec;

    #[test]
    fn queue_preserves_order_and_drains() {
        let mut region = LiveRegion::polite();
        assert!(region.is_empty());
        region.announce("one");
        region.announce("two");
        region.announce("three");
        assert_eq!(region.pending(), 3);

        let drained = region.drain();
        assert_eq!(drained, ["one", "two", "three"]);
        assert!(region.is_empty());
        assert_eq!(region.drain(), Vec::<String>::new());
    }

    #[test]
    fn politeness_is_reported() {
        assert_eq!(LiveRegion::polite().politeness(), Politeness::Polite);
        assert_eq!(LiveRegion::assertive().politeness(), Politeness::Assertive);
        assert!(Politeness::Assertive.interrupts());
        assert!(!Politeness::Polite.interrupts());
    }
}
