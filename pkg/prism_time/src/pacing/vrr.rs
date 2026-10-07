//! §24.3 — variable-refresh-rate (G-Sync/FreeSync) awareness ([`VrrWindow`]).

/// How a desired present interval maps onto a [`VrrWindow`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VrrPresent {
    /// The interval sits inside the panel's variable-refresh window and is
    /// presented as-is.
    InRange {
        /// The present interval (equals the requested interval).
        interval_ns: u64,
    },
    /// The frame is ready faster than the panel's maximum refresh; the present
    /// must wait until the shortest allowed interval has elapsed.
    ClampedFast {
        /// The clamped (minimum) interval the panel can display.
        interval_ns: u64,
    },
    /// The interval is slower than the panel's minimum refresh; Low-Framerate
    /// Compensation duplicates the frame `multiplier` times so the display keeps
    /// refreshing inside its window.
    Lfc {
        /// Number of display refreshes the frame is shown across (`>= 2`).
        multiplier: u32,
        /// The per-refresh sub-interval (`interval / multiplier`), which lies
        /// within the window.
        sub_interval_ns: u64,
    },
}

impl VrrPresent {
    /// The effective per-refresh interval the display actually uses.
    #[inline]
    #[must_use]
    pub const fn effective_interval_ns(&self) -> u64 {
        match *self {
            VrrPresent::InRange { interval_ns } | VrrPresent::ClampedFast { interval_ns } => {
                interval_ns
            }
            VrrPresent::Lfc {
                sub_interval_ns, ..
            } => sub_interval_ns,
        }
    }
}

/// A display's variable-refresh window (`G-Sync` / `FreeSync` / VESA Adaptive-Sync).
///
/// Stored as the fastest and slowest allowed *present intervals* (period =
/// `1 / refresh`). A VRR panel presents whenever a frame is ready, but only
/// inside `[min_interval_ns, max_interval_ns]`: faster frames must wait
/// ([`VrrPresent::ClampedFast`]); frames slower than the window trigger
/// Low-Framerate Compensation ([`VrrPresent::Lfc`]). This lets the fixed-step
/// simulation (§7) run decoupled from the variable display, with the
/// presentation layer's interpolation alpha (§12) absorbing refresh jitter.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VrrWindow {
    min_interval_ns: u64,
    max_interval_ns: u64,
}

impl VrrWindow {
    /// New window from explicit interval bounds (ns). `min_interval_ns` is
    /// clamped to `>= 1`, and `max_interval_ns` to `>= min_interval_ns`, so the
    /// window is always non-empty and ordered.
    #[inline]
    #[must_use]
    pub const fn new(min_interval_ns: u64, max_interval_ns: u64) -> Self {
        let min = if min_interval_ns == 0 {
            1
        } else {
            min_interval_ns
        };
        let max = if max_interval_ns < min {
            min
        } else {
            max_interval_ns
        };
        Self {
            min_interval_ns: min,
            max_interval_ns: max,
        }
    }

    /// New window from a refresh range in whole hertz (e.g. `48..=144`). Both
    /// bounds are clamped to `>= 1` Hz and ordered; the fastest refresh maps to
    /// the shortest interval.
    #[inline]
    #[must_use]
    pub const fn from_hz(min_hz: u32, max_hz: u32) -> Self {
        let lo = if min_hz == 0 { 1 } else { min_hz };
        let hi = if max_hz == 0 { 1 } else { max_hz };
        let (lo, hi) = if lo <= hi { (lo, hi) } else { (hi, lo) };
        // Slowest refresh (lo Hz) -> longest interval == max_interval.
        let max_interval = 1_000_000_000 / lo as u64;
        let min_interval = 1_000_000_000 / hi as u64;
        Self::new(min_interval, max_interval)
    }

    /// Shortest present interval the panel can display (fastest refresh).
    #[inline]
    #[must_use]
    pub const fn min_interval_ns(&self) -> u64 {
        self.min_interval_ns
    }

    /// Longest present interval before Low-Framerate Compensation engages
    /// (slowest refresh).
    #[inline]
    #[must_use]
    pub const fn max_interval_ns(&self) -> u64 {
        self.max_interval_ns
    }

    /// Whether `interval_ns` lies within the variable-refresh window.
    #[inline]
    #[must_use]
    pub const fn contains(&self, interval_ns: u64) -> bool {
        interval_ns >= self.min_interval_ns && interval_ns <= self.max_interval_ns
    }

    /// Map a desired present interval onto the window.
    ///
    /// - inside the window → [`InRange`](VrrPresent::InRange);
    /// - faster than the window → [`ClampedFast`](VrrPresent::ClampedFast) at the
    ///   minimum interval;
    /// - slower than the window → [`Lfc`](VrrPresent::Lfc): the smallest integer
    ///   `multiplier >= 2` whose sub-interval `desired / multiplier` fits at or
    ///   below `max_interval_ns`, keeping the panel refreshing in range.
    #[must_use]
    pub const fn classify(&self, desired_interval_ns: u64) -> VrrPresent {
        if desired_interval_ns < self.min_interval_ns {
            return VrrPresent::ClampedFast {
                interval_ns: self.min_interval_ns,
            };
        }
        if desired_interval_ns <= self.max_interval_ns {
            return VrrPresent::InRange {
                interval_ns: desired_interval_ns,
            };
        }
        // Below the window: ceil-divide to find the smallest multiplier whose
        // sub-interval fits at/under the longest allowed interval.
        // ceil(desired / max_interval) without div_ceil (const-fn friendly).
        let mut multiplier =
            desired_interval_ns.saturating_add(self.max_interval_ns - 1) / self.max_interval_ns;
        if multiplier < 2 {
            multiplier = 2;
        }
        let sub_interval_ns = desired_interval_ns / multiplier;
        VrrPresent::Lfc {
            multiplier: if multiplier > u32::MAX as u64 {
                u32::MAX
            } else {
                multiplier as u32
            },
            sub_interval_ns,
        }
    }

    /// Earliest time the next frame may be presented given it is `ready_ns` and
    /// the previous present was `last_present_ns`: the frame cannot be shown
    /// faster than the panel's maximum refresh, so the present is held to at
    /// least `last_present_ns + min_interval_ns`.
    #[inline]
    #[must_use]
    pub const fn earliest_present_ns(&self, ready_ns: u64, last_present_ns: u64) -> u64 {
        let floor = last_present_ns.saturating_add(self.min_interval_ns);
        if ready_ns > floor {
            ready_ns
        } else {
            floor
        }
    }
}
