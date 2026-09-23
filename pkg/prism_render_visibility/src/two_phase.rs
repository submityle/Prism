use crate::{HistoryPolicy, VisibilityStageMask};

/// Output classification for the previous-HZB early test.
pub fn classify_early_hzb(
    history: HistoryPolicy,
    previous_hzb_occluded: Option<bool>,
) -> VisibilityStageMask {
    if history == HistoryPolicy::Reuse && previous_hzb_occluded == Some(true) {
        VisibilityStageMask::LATE_RETEST
    } else {
        VisibilityStageMask::EARLY
    }
}

/// Resolves the late current-HZB test. Items that were accepted early remain
/// visible; only deferred candidates may be rejected at this stage.
pub fn resolve_current_hzb(
    stages: VisibilityStageMask,
    current_hzb_occluded: Option<bool>,
) -> VisibilityStageMask {
    if stages.contains(VisibilityStageMask::EARLY) {
        return stages | VisibilityStageMask::LATE_VISIBLE;
    }
    if stages.contains(VisibilityStageMask::LATE_RETEST)
        && current_hzb_occluded != Some(true)
    {
        return stages | VisibilityStageMask::LATE_VISIBLE;
    }
    stages
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn camera_cut_and_missing_history_stay_in_early_set() {
        assert_eq!(
            classify_early_hzb(HistoryPolicy::Reset, Some(true)),
            VisibilityStageMask::EARLY
        );
        assert_eq!(
            classify_early_hzb(HistoryPolicy::Conservative, Some(true)),
            VisibilityStageMask::EARLY
        );
        assert_eq!(
            classify_early_hzb(HistoryPolicy::Reuse, None),
            VisibilityStageMask::EARLY
        );
    }

    #[test]
    fn previous_occlusion_is_retested_against_current_hzb() {
        let deferred = classify_early_hzb(HistoryPolicy::Reuse, Some(true));
        assert_eq!(deferred, VisibilityStageMask::LATE_RETEST);
        assert!(resolve_current_hzb(deferred, Some(false))
            .contains(VisibilityStageMask::LATE_VISIBLE));
        assert!(!resolve_current_hzb(deferred, Some(true))
            .contains(VisibilityStageMask::LATE_VISIBLE));
    }
}
