//! Per-system QoS lane annotation (design §24.1).
//!
//! The ECS conflict-graph executor (§8.2) and fiber job graph (§8.3)
//! deliberately **do not** own a thread pool or a priority policy of their own.
//! Instead a system declares *how urgent* it is with a [`SystemLane`], and the
//! multi-threaded executor maps that annotation onto a `prism_tasks` priority
//! lane when it dispatches the system's body. This keeps a single source of
//! truth: scheduling/priority mechanics live in `prism_tasks`, while the ECS
//! only declares «access set + order + lane annotation».
//!
//! The three lanes mirror the design's own vocabulary verbatim:
//!
//! * [`SystemLane::Critical`] — the critical path (physics, extract, transform
//!   propagation): must always run this frame, ahead of everything else.
//! * [`SystemLane::Normal`] — ordinary gameplay work; the default.
//! * [`SystemLane::Background`] — deferrable async precompute (prefetch, idle
//!   bakes) that a frame-budget scheduler may push to a later frame.
//!
//! The annotation is a pure, `no_std` value with no dependency on
//! `prism_tasks`; the mapping to a concrete [`prism_tasks::Priority`] is only
//! compiled in under the `multi_thread` feature (where the pool exists). This
//! lets the base kernel carry lane metadata — e.g. for diagnostics or a custom
//! executor — without pulling in the scheduler.

/// The quality-of-service lane a system is dispatched on (design §24.1).
///
/// Ordered by urgency: `Background < Normal < Critical`. [`Default`] is
/// [`SystemLane::Normal`], so an unannotated system behaves exactly as before
/// this annotation existed.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default)]
pub enum SystemLane {
    /// Deferrable background work (prefetch, idle bakes); a frame-budget
    /// scheduler may defer it to a later frame. Maps to
    /// [`prism_tasks::Priority::Background`].
    Background,
    /// Ordinary gameplay work. The default lane; maps to
    /// [`prism_tasks::Priority::Normal`].
    #[default]
    Normal,
    /// The critical path (physics / extract / transform propagation) that must
    /// complete this frame ahead of everything else. Maps to
    /// [`prism_tasks::Priority::Critical`].
    Critical,
}

impl SystemLane {
    /// Maps the lane onto the `prism_tasks` scheduling priority the
    /// multi-threaded executor dispatches it at (design §24.1).
    ///
    /// Only compiled with the `multi_thread` feature, where the pool and its
    /// [`prism_tasks::Priority`] type exist.
    #[cfg(feature = "multi_thread")]
    #[inline]
    #[must_use]
    pub fn to_priority(self) -> prism_tasks::Priority {
        match self {
            SystemLane::Background => prism_tasks::Priority::Background,
            SystemLane::Normal => prism_tasks::Priority::Normal,
            SystemLane::Critical => prism_tasks::Priority::Critical,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::SystemLane;

    #[test]
    fn default_is_normal() {
        assert_eq!(SystemLane::default(), SystemLane::Normal);
    }

    #[test]
    fn lanes_are_ordered_by_urgency() {
        assert!(SystemLane::Background < SystemLane::Normal);
        assert!(SystemLane::Normal < SystemLane::Critical);
    }

    #[cfg(feature = "multi_thread")]
    #[test]
    fn maps_each_lane_to_its_priority() {
        use prism_tasks::Priority;
        assert_eq!(SystemLane::Background.to_priority(), Priority::Background);
        assert_eq!(SystemLane::Normal.to_priority(), Priority::Normal);
        assert_eq!(SystemLane::Critical.to_priority(), Priority::Critical);
        // Priority ordering must agree with lane ordering so a saturated pool
        // drains Critical ahead of Normal ahead of Background.
        assert!(SystemLane::Background.to_priority() < SystemLane::Critical.to_priority());
    }
}
