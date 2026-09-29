//! Residency budget accounting.
//!
//! A [`ResidencyBudget`] describes three independent limits that gate what the
//! scheduler may keep and upload in one frame:
//!
//! * `soft_bytes` — a comfort target. Staying at or under it means the pool is
//!   not under memory pressure; exceeding it (but not `hard_bytes`) is allowed
//!   but flagged so higher layers can throttle demand.
//! * `hard_bytes` — the physical ceiling. The chosen resident set may never
//!   exceed it; admission stops here.
//! * `upload_bytes_per_frame` — how many bytes of *new* residency the backend
//!   can stream in a single frame. Already-resident bytes never count against
//!   it; only fresh uploads do.
//!
//! [`BudgetLedger`] is the running tally the scheduler threads through a single
//! admission pass. All arithmetic is checked integer math, so overflow can never
//! silently admit past a ceiling, and results are exact and deterministic. No
//! byte here is a real allocation: the backend performs the uploads, pending the
//! `GPU` backend.

use super::ResidencyBudget;

impl ResidencyBudget {
    /// Builds a budget from its three limits.
    ///
    /// `soft_bytes` is clamped up to `hard_bytes` if a caller passes a soft
    /// target above the hard ceiling, keeping `soft <= hard` an invariant the
    /// rest of the module can rely on.
    #[must_use]
    pub const fn new(soft_bytes: u64, hard_bytes: u64, upload_bytes_per_frame: u64) -> Self {
        let soft_bytes = if soft_bytes > hard_bytes {
            hard_bytes
        } else {
            soft_bytes
        };
        Self {
            soft_bytes,
            hard_bytes,
            upload_bytes_per_frame,
        }
    }

    /// Whether `bytes` fits within the hard ceiling.
    #[must_use]
    pub const fn within_hard(&self, bytes: u64) -> bool {
        bytes <= self.hard_bytes
    }

    /// Whether `bytes` fits within the soft comfort target.
    #[must_use]
    pub const fn within_soft(&self, bytes: u64) -> bool {
        bytes <= self.soft_bytes
    }

    /// Bytes of headroom left under the hard ceiling above `used`.
    #[must_use]
    pub const fn hard_headroom(&self, used: u64) -> u64 {
        self.hard_bytes.saturating_sub(used)
    }
}

/// Running byte tally for one scheduling pass.
///
/// Tracks the bytes committed to the resident set and the bytes uploaded this
/// frame, and answers the two admission questions the scheduler asks: *would
/// keeping this resident stay under the hard ceiling?* and *would uploading this
/// fresh resource stay under the per-frame upload limit?* The ledger only ever
/// grows within a pass; a new frame starts a new ledger.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BudgetLedger {
    budget: ResidencyBudget,
    resident_bytes: u64,
    uploaded_bytes: u64,
}

impl BudgetLedger {
    /// Starts an empty ledger for `budget`.
    #[must_use]
    pub const fn new(budget: ResidencyBudget) -> Self {
        Self {
            budget,
            resident_bytes: 0,
            uploaded_bytes: 0,
        }
    }

    /// Bytes currently committed to the resident set.
    #[must_use]
    pub const fn resident_bytes(&self) -> u64 {
        self.resident_bytes
    }

    /// Bytes of fresh residency uploaded so far this frame.
    #[must_use]
    pub const fn uploaded_bytes(&self) -> u64 {
        self.uploaded_bytes
    }

    /// Whether the resident set currently exceeds the soft target.
    #[must_use]
    pub const fn over_soft(&self) -> bool {
        self.resident_bytes > self.budget.soft_bytes
    }

    /// Whether committing `byte_cost` more resident bytes stays under the hard
    /// ceiling. `already_resident` resources do not consume upload budget, but
    /// they still count toward the hard ceiling like anything else.
    #[must_use]
    pub fn can_admit(&self, byte_cost: u64, already_resident: bool) -> bool {
        let Some(next_resident) = self.resident_bytes.checked_add(byte_cost) else {
            return false;
        };
        if !self.budget.within_hard(next_resident) {
            return false;
        }
        if already_resident {
            return true;
        }
        let Some(next_upload) = self.uploaded_bytes.checked_add(byte_cost) else {
            return false;
        };
        next_upload <= self.budget.upload_bytes_per_frame
    }

    /// Commits `byte_cost` to the resident set, counting an upload unless the
    /// resource was `already_resident`.
    ///
    /// Returns `true` when the commit succeeded. A commit that would breach the
    /// hard ceiling or the per-frame upload limit is refused and leaves the
    /// ledger unchanged, so a failed admission never partially charges.
    pub fn commit(&mut self, byte_cost: u64, already_resident: bool) -> bool {
        if !self.can_admit(byte_cost, already_resident) {
            return false;
        }
        self.resident_bytes += byte_cost;
        if !already_resident {
            self.uploaded_bytes += byte_cost;
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn budget() -> ResidencyBudget {
        ResidencyBudget::new(1_000, 2_000, 500)
    }

    #[test]
    fn new_clamps_soft_below_hard() {
        let b = ResidencyBudget::new(9_000, 2_000, 100);
        assert_eq!(b.soft_bytes, 2_000);
        assert_eq!(b.hard_bytes, 2_000);
    }

    #[test]
    fn resident_admission_respects_hard_ceiling() {
        let mut ledger = BudgetLedger::new(budget());
        // Resident bytes ignore the upload limit but honour the hard ceiling.
        assert!(ledger.commit(1_500, true));
        assert!(ledger.commit(500, true));
        assert_eq!(ledger.resident_bytes(), 2_000);
        // 2_000 is the ceiling; one more byte is refused.
        assert!(!ledger.commit(1, true));
        assert_eq!(ledger.resident_bytes(), 2_000);
        assert_eq!(ledger.uploaded_bytes(), 0);
    }

    #[test]
    fn fresh_uploads_respect_per_frame_limit() {
        let mut ledger = BudgetLedger::new(budget());
        assert!(ledger.commit(400, false));
        // Another 400 would push uploads to 800 > 500 limit, even though the
        // hard ceiling has plenty of room.
        assert!(!ledger.commit(400, false));
        assert!(ledger.commit(100, false));
        assert_eq!(ledger.uploaded_bytes(), 500);
        assert_eq!(ledger.resident_bytes(), 500);
    }

    #[test]
    fn over_soft_flag_tracks_comfort_target() {
        let mut ledger = BudgetLedger::new(budget());
        assert!(ledger.commit(1_000, true));
        assert!(!ledger.over_soft());
        assert!(ledger.commit(1, true));
        assert!(ledger.over_soft());
    }

    #[test]
    fn commit_overflow_is_refused() {
        let mut ledger = BudgetLedger::new(ResidencyBudget::new(u64::MAX, u64::MAX, u64::MAX));
        assert!(ledger.commit(u64::MAX, true));
        // Adding one more would overflow u64; must be refused, not wrap.
        assert!(!ledger.commit(1, true));
    }

    #[test]
    fn headroom_saturates() {
        let b = budget();
        assert_eq!(b.hard_headroom(500), 1_500);
        assert_eq!(b.hard_headroom(9_999), 0);
    }
}
