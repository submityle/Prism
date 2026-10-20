use crate::command::ActorId;
use core::fmt;

/// A deterministic hybrid logical timestamp.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Hash)]
pub struct HlcTimestamp {
    /// Physical time supplied by the caller. The clock never reads wall time.
    pub physical: u64,
    /// Logical counter used when physical time does not advance.
    pub logical: u32,
}

impl HlcTimestamp {
    /// The initial timestamp.
    pub const ZERO: Self = Self {
        physical: 0,
        logical: 0,
    };

    /// Constructs a timestamp from explicit wire values.
    #[must_use]
    pub const fn new(physical: u64, logical: u32) -> Self {
        Self { physical, logical }
    }
}

/// Errors produced by clock transitions.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ClockError {
    /// The logical counter cannot be advanced without overflowing.
    LogicalOverflow,
}

impl fmt::Display for ClockError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LogicalOverflow => formatter.write_str("HLC logical counter overflowed"),
        }
    }
}

impl std::error::Error for ClockError {}

/// A caller-driven HLC state machine.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HlcClock {
    actor: ActorId,
    last: HlcTimestamp,
}

impl HlcClock {
    /// Creates a clock for an actor.
    #[must_use]
    pub fn new(actor: ActorId) -> Self {
        Self {
            actor,
            last: HlcTimestamp::ZERO,
        }
    }

    /// Returns the actor owning this clock.
    #[must_use]
    pub fn actor(&self) -> &ActorId {
        &self.actor
    }

    /// Returns the last emitted or observed timestamp.
    #[must_use]
    pub const fn last(&self) -> HlcTimestamp {
        self.last
    }

    /// Emits a local timestamp using caller-provided physical time.
    pub fn tick(&mut self, physical: u64) -> Result<HlcTimestamp, ClockError> {
        if physical > self.last.physical {
            self.last = HlcTimestamp::new(physical, 0);
        } else {
            self.last.logical = self
                .last
                .logical
                .checked_add(1)
                .ok_or(ClockError::LogicalOverflow)?;
        }
        Ok(self.last)
    }

    /// Observes a remote timestamp and advances the local clock.
    pub fn observe(
        &mut self,
        remote: HlcTimestamp,
        physical: u64,
    ) -> Result<HlcTimestamp, ClockError> {
        let next_physical = self.last.physical.max(remote.physical).max(physical);
        let next_logical =
            if next_physical == self.last.physical && next_physical == remote.physical {
                self.last.logical.max(remote.logical)
            } else if next_physical == self.last.physical {
                self.last.logical
            } else if next_physical == remote.physical {
                remote.logical
            } else {
                0
            };
        self.last = HlcTimestamp::new(next_physical, next_logical);
        self.tick(next_physical)
    }
}
