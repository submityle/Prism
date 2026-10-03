//! The fixed main-schedule phases (design §8.2).
//!
//! Every system belongs to exactly one [`Phase`]. The phases run in a fixed
//! order — `First → PreUpdate → Update → FixedUpdate → PostUpdate → Last` — and
//! the schedule seeds the equivalent chain of set-ordering edges automatically,
//! so a system in an earlier phase is always ordered before one in a later
//! phase regardless of how the two were added.

use crate::schedule::set::{SystemSet, SystemSetId};

/// A fixed slot in the main schedule. Systems default to [`Phase::Update`].
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub enum Phase {
    /// Runs first: input intake, frame bookkeeping.
    First,
    /// Pre-update: before gameplay logic (e.g. time/transform propagation in).
    PreUpdate,
    /// The default phase for ordinary gameplay systems.
    Update,
    /// Fixed-timestep simulation work (deterministic step; see design §14).
    FixedUpdate,
    /// Post-update: after gameplay, before frame end (e.g. transform out).
    PostUpdate,
    /// Runs last: cleanup, frame finalisation, extract hand-off.
    Last,
}

impl Phase {
    /// All phases in their canonical execution order.
    pub const ORDER: [Phase; 6] = [
        Phase::First,
        Phase::PreUpdate,
        Phase::Update,
        Phase::FixedUpdate,
        Phase::PostUpdate,
        Phase::Last,
    ];

    /// This phase's position in [`Phase::ORDER`].
    #[inline]
    #[must_use]
    pub fn index(self) -> u64 {
        self as u64
    }
}

impl Default for Phase {
    #[inline]
    fn default() -> Self {
        Phase::Update
    }
}

impl SystemSet for Phase {
    #[inline]
    fn set_id(&self) -> SystemSetId {
        SystemSetId::with::<Phase>(self.index())
    }
}
