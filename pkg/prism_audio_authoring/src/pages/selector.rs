//! Hysteresis and hot-switch rate limiting for page selection.
//!
//! # Provenance
//! Original work; no Unreal Engine, Unity, Godot, Wwise, FMOD, Steam Audio, or
//! Google Resonance Audio source or derived code; no AI/ML. Debouncing a
//! fluctuating control signal with a dwell count and a minimum switch interval
//! is classical control hygiene; this is an independent implementation applied
//! to page selection.
//!
//! # Relationship
//! Resolves the design section 42 concern that coupling "`MetaSound` Pages-style
//! tiered compilation" to the design section 32 quality governor can thrash:
//! when the governor's quality tier dithers around a page boundary, naively
//! recompiling on every change would cause a compile storm. [`PageSelector`]
//! operates on *resolved page indices* (so jitter that stays within one page
//! never churns), requires a change to persist for `confirm_ticks` observations
//! (hysteresis), and enforces at least `min_switch_gap` ticks between committed
//! switches (a hot-switch rate cap). It never allocates and is deterministic,
//! so it is safe to run on a control thread alongside the design section 21
//! command ring. The committed switch is the signal for a host to recompile the
//! resolved [`super::PagedPatch`] page.

use super::paged_patch::PagedPatch;
use super::tier::QualityLevel;

/// Why a requested page change has not yet been committed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeferReason {
    /// The change has not persisted for `confirm_ticks` observations yet.
    SettlingHysteresis,
    /// The change is confirmed but the minimum switch interval has not elapsed.
    RateLimited,
}

/// The outcome of one [`PageSelector::observe`] call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PageDecision {
    /// The request resolves to the already-active page; nothing to do.
    Unchanged,
    /// A different page is requested but not yet committed to.
    Deferred {
        /// Index of the page the request currently resolves to.
        target_index: usize,
        /// Why the switch is being held back.
        reason: DeferReason,
    },
    /// The active page changed; the caller should recompile `to_index`.
    Switched {
        /// Index of the page that was active before this call.
        from_index: usize,
        /// Index of the newly active page to recompile.
        to_index: usize,
    },
}

/// Debounced, rate-limited selection of a [`PagedPatch`] page.
///
/// Feed each control-rate observation to [`PageSelector::observe`] with the
/// current requested [`QualityLevel`] and a monotonically non-decreasing tick.
/// The selector returns [`PageDecision::Switched`] only when a genuine,
/// settled, rate-permitted page change occurs.
#[derive(Debug, Clone)]
pub struct PageSelector {
    /// Index of the currently committed page.
    active_index: usize,
    /// Observations a differing target must persist before committing.
    confirm_ticks: u32,
    /// Minimum tick delta between two committed switches.
    min_switch_gap: u64,
    /// The candidate target and how many consecutive observations backed it.
    pending: Option<Pending>,
    /// Tick of the most recent committed switch, if any.
    last_switch_tick: Option<u64>,
}

/// A candidate page change accumulating observations.
#[derive(Debug, Clone, Copy)]
struct Pending {
    /// Index of the candidate page.
    target_index: usize,
    /// Consecutive observations backing this candidate so far.
    count: u32,
}

impl PageSelector {
    /// Builds a selector starting on `active_index`.
    ///
    /// `confirm_ticks` is the hysteresis dwell: a differing target must be
    /// observed this many times in a row before it commits (`0` and `1` both
    /// commit on the first differing observation). `min_switch_gap` is the
    /// minimum number of ticks between two committed switches.
    #[must_use]
    pub fn new(active_index: usize, confirm_ticks: u32, min_switch_gap: u64) -> Self {
        Self {
            active_index,
            confirm_ticks,
            min_switch_gap,
            pending: None,
            last_switch_tick: None,
        }
    }

    /// Builds a selector whose active page is the one serving `level`.
    #[must_use]
    pub fn for_level(
        paged: &PagedPatch,
        level: QualityLevel,
        confirm_ticks: u32,
        min_switch_gap: u64,
    ) -> Self {
        Self::new(paged.resolve_index(level), confirm_ticks, min_switch_gap)
    }

    /// Returns the index of the currently committed page.
    #[must_use]
    pub fn active_index(&self) -> usize {
        self.active_index
    }

    /// Returns the candidate page index currently accumulating confirmation.
    #[must_use]
    pub fn pending_target(&self) -> Option<usize> {
        self.pending.map(|p| p.target_index)
    }

    /// Observes the current requested level at tick `now`.
    ///
    /// Resolves `requested` to a page index on `paged`, then applies hysteresis
    /// and the switch-rate cap to decide whether to commit a change.
    pub fn observe(
        &mut self,
        paged: &PagedPatch,
        requested: QualityLevel,
        now: u64,
    ) -> PageDecision {
        let target_index = paged.resolve_index(requested);
        if target_index == self.active_index {
            self.pending = None;
            return PageDecision::Unchanged;
        }

        let count = match self.pending {
            Some(p) if p.target_index == target_index => p.count.saturating_add(1),
            _ => 1,
        };
        self.pending = Some(Pending {
            target_index,
            count,
        });

        if count < self.confirm_ticks {
            return PageDecision::Deferred {
                target_index,
                reason: DeferReason::SettlingHysteresis,
            };
        }

        if let Some(last) = self.last_switch_tick
            && now.saturating_sub(last) < self.min_switch_gap
        {
            return PageDecision::Deferred {
                target_index,
                reason: DeferReason::RateLimited,
            };
        }

        let from_index = self.active_index;
        self.active_index = target_index;
        self.last_switch_tick = Some(now);
        self.pending = None;
        PageDecision::Switched {
            from_index,
            to_index: target_index,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pages::page::PatchPage;
    use crate::patch::PatchDescription;
    use alloc::string::ToString;
    use alloc::vec;

    fn ladder() -> PagedPatch {
        PagedPatch::new(vec![
            PatchPage::new(QualityLevel::new(0), "base".to_string(), PatchDescription::new()),
            PatchPage::new(QualityLevel::new(2), "high".to_string(), PatchDescription::new()),
            PatchPage::new(QualityLevel::new(4), "ultra".to_string(), PatchDescription::new()),
        ])
        .expect("valid ladder")
    }

    #[test]
    fn jitter_within_a_page_never_switches() {
        let paged = ladder();
        let mut sel = PageSelector::for_level(&paged, QualityLevel::new(0), 2, 0);
        // Levels 0 and 1 both resolve to the base page: no churn.
        assert_eq!(
            sel.observe(&paged, QualityLevel::new(1), 10),
            PageDecision::Unchanged
        );
        assert_eq!(
            sel.observe(&paged, QualityLevel::new(0), 11),
            PageDecision::Unchanged
        );
        assert_eq!(sel.active_index(), 0);
    }

    #[test]
    fn hysteresis_requires_sustained_request() {
        let paged = ladder();
        let mut sel = PageSelector::new(0, 3, 0);
        // First two observations are deferred while confirmation accrues.
        assert_eq!(
            sel.observe(&paged, QualityLevel::new(2), 0),
            PageDecision::Deferred {
                target_index: 1,
                reason: DeferReason::SettlingHysteresis,
            }
        );
        assert_eq!(
            sel.observe(&paged, QualityLevel::new(2), 1),
            PageDecision::Deferred {
                target_index: 1,
                reason: DeferReason::SettlingHysteresis,
            }
        );
        // Third sustained observation commits.
        assert_eq!(
            sel.observe(&paged, QualityLevel::new(2), 2),
            PageDecision::Switched {
                from_index: 0,
                to_index: 1,
            }
        );
        assert_eq!(sel.active_index(), 1);
    }

    #[test]
    fn interrupted_request_resets_confirmation() {
        let paged = ladder();
        let mut sel = PageSelector::new(0, 2, 0);
        let _ = sel.observe(&paged, QualityLevel::new(2), 0);
        // Falling back to the active page clears the pending candidate.
        assert_eq!(
            sel.observe(&paged, QualityLevel::new(0), 1),
            PageDecision::Unchanged
        );
        assert_eq!(sel.pending_target(), None);
        // The candidate must build confirmation again from scratch.
        assert!(matches!(
            sel.observe(&paged, QualityLevel::new(2), 2),
            PageDecision::Deferred { .. }
        ));
    }

    #[test]
    fn rate_cap_defers_rapid_second_switch() {
        let paged = ladder();
        let mut sel = PageSelector::new(0, 0, 100);
        // Immediate confirm (confirm_ticks 0) commits the first switch.
        assert_eq!(
            sel.observe(&paged, QualityLevel::new(2), 10),
            PageDecision::Switched {
                from_index: 0,
                to_index: 1,
            }
        );
        // A second switch within the gap is rate limited.
        assert_eq!(
            sel.observe(&paged, QualityLevel::new(4), 50),
            PageDecision::Deferred {
                target_index: 2,
                reason: DeferReason::RateLimited,
            }
        );
        // Once the gap elapses it commits.
        assert_eq!(
            sel.observe(&paged, QualityLevel::new(4), 110),
            PageDecision::Switched {
                from_index: 1,
                to_index: 2,
            }
        );
    }
}
