//! Indirect command counting and budgeting.
//!
//! GPU-generated work is emitted as indirect commands: a compute pass writes an
//! indirect buffer, and a later `draw_indirect_count` / `dispatch_indirect`
//! consumes some prefix of it. The `CPU` cannot know the exact count a shader
//! produced, but it does size the indirect buffers and the count buffers up
//! front, and those sizes are hard caps. Exceeding them is undefined behavior on
//! the GPU, so the count is clamped by `maxDrawCount` regardless.
//!
//! This module owns the `CPU`-side budget for those caps. It separates draw and
//! dispatch commands (they live in distinct indirect buffers), tallies how many
//! of each have been reserved, and reports when a reservation would exceed the
//! budget. As with queues, a budget can be lossy (clamp and count the surplus)
//! or fatal (reject the reservation outright).
//!
//! Writing the indirect and count buffers and issuing the indirect commands are
//! pending the GPU backend; here we only reason about the counts.

use super::ratio::Ratio;

/// Which indirect command stream a reservation targets.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum IndirectKind {
    /// `vkCmdDrawIndirectCount` and friends: graphics draws.
    Draw,
    /// `vkCmdDispatchIndirect`: compute dispatches.
    Dispatch,
}

/// Per-stream caps on indirect command counts.
///
/// Each cap is the maximum number of commands the corresponding indirect buffer
/// was sized for. When `fatal_on_overflow` is set, a reservation past a cap is
/// rejected; otherwise the surplus is clamped away and counted as overflow.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct IndirectBudget {
    pub max_draws: u32,
    pub max_dispatches: u32,
    pub fatal_on_overflow: bool,
}

impl IndirectBudget {
    /// A lossy budget with the given per-stream caps.
    #[must_use]
    pub const fn lossy(max_draws: u32, max_dispatches: u32) -> Self {
        Self {
            max_draws,
            max_dispatches,
            fatal_on_overflow: false,
        }
    }

    /// A fatal budget with the given per-stream caps.
    #[must_use]
    pub const fn fatal(max_draws: u32, max_dispatches: u32) -> Self {
        Self {
            max_draws,
            max_dispatches,
            fatal_on_overflow: true,
        }
    }

    /// The cap for a given command stream.
    #[must_use]
    pub const fn cap(self, kind: IndirectKind) -> u32 {
        match kind {
            IndirectKind::Draw => self.max_draws,
            IndirectKind::Dispatch => self.max_dispatches,
        }
    }
}

/// Reservation rejected because a fatal budget would be exceeded.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct BudgetExceeded {
    /// Which stream overflowed.
    pub kind: IndirectKind,
    /// Commands requested by the failing reservation.
    pub requested: u32,
    /// Commands that could have been reserved before hitting the cap.
    pub available: u32,
}

/// Outcome of a successful reservation on a lossy or fatal budget.
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct ReserveOutcome {
    /// Commands admitted within budget.
    pub reserved: u32,
    /// Commands clamped away for exceeding the cap (lossy budgets only).
    pub clamped: u32,
}

/// Running tally of reserved indirect commands, gated by an [`IndirectBudget`].
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq)]
pub struct IndirectTally {
    budget: IndirectBudget,
    draws: u32,
    dispatches: u32,
    clamped_draws: u32,
    clamped_dispatches: u32,
}

impl IndirectTally {
    /// Creates an empty tally bound to `budget`.
    #[must_use]
    pub const fn new(budget: IndirectBudget) -> Self {
        Self {
            budget,
            draws: 0,
            dispatches: 0,
            clamped_draws: 0,
            clamped_dispatches: 0,
        }
    }

    /// The budget this tally is measured against.
    #[must_use]
    pub const fn budget(self) -> IndirectBudget {
        self.budget
    }

    /// Commands currently reserved on `kind`.
    #[must_use]
    pub const fn reserved(self, kind: IndirectKind) -> u32 {
        match kind {
            IndirectKind::Draw => self.draws,
            IndirectKind::Dispatch => self.dispatches,
        }
    }

    /// Commands clamped away on `kind` over this tally's lifetime.
    #[must_use]
    pub const fn clamped(self, kind: IndirectKind) -> u32 {
        match kind {
            IndirectKind::Draw => self.clamped_draws,
            IndirectKind::Dispatch => self.clamped_dispatches,
        }
    }

    /// Remaining headroom on `kind` before the cap is reached.
    #[must_use]
    pub const fn remaining(self, kind: IndirectKind) -> u32 {
        self.budget.cap(kind) - self.reserved(kind)
    }

    /// Fill fraction of `kind`: reserved over cap, as an exact ratio.
    #[must_use]
    pub fn fill(self, kind: IndirectKind) -> Ratio {
        Ratio::new(
            u64::from(self.reserved(kind)),
            u64::from(self.budget.cap(kind)),
        )
    }

    /// Reserves `count` commands on `kind`.
    ///
    /// A lossy budget clamps to the remaining headroom and counts the surplus; a
    /// fatal budget returns [`BudgetExceeded`] and leaves the tally unchanged.
    pub fn reserve(
        &mut self,
        kind: IndirectKind,
        count: u32,
    ) -> Result<ReserveOutcome, BudgetExceeded> {
        let available = self.remaining(kind);
        if count <= available {
            self.add_reserved(kind, count);
            return Ok(ReserveOutcome {
                reserved: count,
                clamped: 0,
            });
        }

        if self.budget.fatal_on_overflow {
            return Err(BudgetExceeded {
                kind,
                requested: count,
                available,
            });
        }

        let clamped = count - available;
        self.add_reserved(kind, available);
        self.add_clamped(kind, clamped);
        Ok(ReserveOutcome {
            reserved: available,
            clamped,
        })
    }

    /// Releases up to `count` reserved commands on `kind`, e.g. after a frame's
    /// indirect buffer is recycled. Returns the number released.
    pub fn release(&mut self, kind: IndirectKind, count: u32) -> u32 {
        let released = count.min(self.reserved(kind));
        match kind {
            IndirectKind::Draw => self.draws -= released,
            IndirectKind::Dispatch => self.dispatches -= released,
        }
        released
    }

    fn add_reserved(&mut self, kind: IndirectKind, count: u32) {
        match kind {
            IndirectKind::Draw => self.draws += count,
            IndirectKind::Dispatch => self.dispatches += count,
        }
    }

    fn add_clamped(&mut self, kind: IndirectKind, count: u32) {
        match kind {
            IndirectKind::Draw => {
                self.clamped_draws = self.clamped_draws.saturating_add(count);
            }
            IndirectKind::Dispatch => {
                self.clamped_dispatches = self.clamped_dispatches.saturating_add(count);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reserve_within_budget() {
        let mut t = IndirectTally::new(IndirectBudget::lossy(100, 50));
        let out = t.reserve(IndirectKind::Draw, 40).unwrap();
        assert_eq!(
            out,
            ReserveOutcome {
                reserved: 40,
                clamped: 0
            }
        );
        assert_eq!(t.reserved(IndirectKind::Draw), 40);
        assert_eq!(t.remaining(IndirectKind::Draw), 60);
        assert_eq!(t.reserved(IndirectKind::Dispatch), 0);
    }

    #[test]
    fn streams_are_independent() {
        let mut t = IndirectTally::new(IndirectBudget::lossy(10, 4));
        t.reserve(IndirectKind::Draw, 10).unwrap();
        let out = t.reserve(IndirectKind::Dispatch, 3).unwrap();
        assert_eq!(out.reserved, 3);
        assert_eq!(t.remaining(IndirectKind::Draw), 0);
        assert_eq!(t.remaining(IndirectKind::Dispatch), 1);
    }

    #[test]
    fn lossy_clamps_and_counts() {
        let mut t = IndirectTally::new(IndirectBudget::lossy(8, 8));
        let out = t.reserve(IndirectKind::Draw, 20).unwrap();
        assert_eq!(
            out,
            ReserveOutcome {
                reserved: 8,
                clamped: 12
            }
        );
        assert_eq!(t.clamped(IndirectKind::Draw), 12);
        assert_eq!(t.fill(IndirectKind::Draw), Ratio::ONE);
    }

    #[test]
    fn fatal_rejects_without_mutation() {
        let mut t = IndirectTally::new(IndirectBudget::fatal(8, 8));
        t.reserve(IndirectKind::Draw, 5).unwrap();
        let before = t;
        let err = t.reserve(IndirectKind::Draw, 5).unwrap_err();
        assert_eq!(
            err,
            BudgetExceeded {
                kind: IndirectKind::Draw,
                requested: 5,
                available: 3,
            }
        );
        assert_eq!(t, before);
    }

    #[test]
    fn zero_cap_stream_is_always_full() {
        let mut t = IndirectTally::new(IndirectBudget::lossy(0, 4));
        let out = t.reserve(IndirectKind::Draw, 3).unwrap();
        assert_eq!(
            out,
            ReserveOutcome {
                reserved: 0,
                clamped: 3
            }
        );
        assert_eq!(t.fill(IndirectKind::Draw), Ratio::ZERO);
    }

    #[test]
    fn release_frees_headroom() {
        let mut t = IndirectTally::new(IndirectBudget::lossy(10, 10));
        t.reserve(IndirectKind::Dispatch, 7).unwrap();
        assert_eq!(t.release(IndirectKind::Dispatch, 3), 3);
        assert_eq!(t.reserved(IndirectKind::Dispatch), 4);
        // Releasing more than reserved is clamped.
        assert_eq!(t.release(IndirectKind::Dispatch, 99), 4);
        assert_eq!(t.reserved(IndirectKind::Dispatch), 0);
    }

    #[test]
    fn clamped_counter_saturates() {
        let mut t = IndirectTally::new(IndirectBudget::lossy(0, 0));
        t.clamped_draws = u32::MAX - 1;
        t.reserve(IndirectKind::Draw, 10).unwrap();
        assert_eq!(t.clamped(IndirectKind::Draw), u32::MAX);
    }

    #[test]
    fn determinism_same_sequence() {
        let run = || {
            let mut t = IndirectTally::new(IndirectBudget::lossy(16, 16));
            t.reserve(IndirectKind::Draw, 10).unwrap();
            t.reserve(IndirectKind::Dispatch, 4).unwrap();
            t.release(IndirectKind::Draw, 5);
            t.reserve(IndirectKind::Draw, 20).unwrap();
            t
        };
        assert_eq!(run(), run());
    }
}
