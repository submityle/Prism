//! Priority lanes for interruptible reconciliation.
//!
//! Loom borrows React Fiber's *lane* model: every unit of work carries a
//! [`Lane`] that ranks how urgently it must land. The scheduler always drains
//! the highest-priority non-empty lane first, so a freshly-enqueued input
//! response preempts an in-flight off-screen list build without discarding the
//! partial progress already made on the lower lane.
//!
//! Lanes are a small, totally-ordered set rather than React's 31-bit mask: a
//! game UI only needs a handful of coarse urgency classes, and a dense enum
//! keeps selection branch-free and the public API self-describing.

use core::fmt;

/// A reconciliation urgency class, ordered from most to least urgent.
///
/// `Ord` ranks lanes by urgency: [`Lane::Immediate`] is the greatest, so
/// `max()` over a set of pending lanes yields the one to service next.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Lane {
    /// Off-screen, speculative warm-up work; yields to everything else.
    Idle,
    /// Content scrolled outside the viewport (overscan / prefetch band).
    Offscreen,
    /// Content inside the visible viewport that is not yet painted.
    Visible,
    /// Running transitions and animation-driven updates.
    Animation,
    /// Direct responses to pointer / keyboard input.
    Input,
    /// Synchronous, must-land-this-frame work (focus, text caret).
    Immediate,
}

impl Lane {
    /// Every lane, ordered most urgent first.
    pub const ALL: [Lane; 6] = [
        Lane::Immediate,
        Lane::Input,
        Lane::Animation,
        Lane::Visible,
        Lane::Offscreen,
        Lane::Idle,
    ];

    /// The number of distinct lanes.
    pub const COUNT: usize = Self::ALL.len();

    /// A dense `0..COUNT` index with `0` = [`Lane::Immediate`] (most urgent).
    ///
    /// Stable across releases so callers may key fixed-size arrays by lane.
    #[must_use]
    pub const fn index(self) -> usize {
        match self {
            Lane::Immediate => 0,
            Lane::Input => 1,
            Lane::Animation => 2,
            Lane::Visible => 3,
            Lane::Offscreen => 4,
            Lane::Idle => 5,
        }
    }

    /// Rebuilds a [`Lane`] from the [`index`](Lane::index) encoding.
    ///
    /// Returns [`None`] when `index >= COUNT`, so decoding untrusted input
    /// degrades to a skipped lane instead of a trap.
    #[must_use]
    pub const fn from_index(index: usize) -> Option<Lane> {
        match index {
            0 => Some(Lane::Immediate),
            1 => Some(Lane::Input),
            2 => Some(Lane::Animation),
            3 => Some(Lane::Visible),
            4 => Some(Lane::Offscreen),
            5 => Some(Lane::Idle),
            _ => None,
        }
    }

    /// Whether `self` must be serviced before `other`.
    #[must_use]
    pub const fn preempts(self, other: Lane) -> bool {
        self.index() < other.index()
    }
}

impl fmt::Display for Lane {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Lane::Immediate => "immediate",
            Lane::Input => "input",
            Lane::Animation => "animation",
            Lane::Visible => "visible",
            Lane::Offscreen => "offscreen",
            Lane::Idle => "idle",
        };
        f.write_str(name)
    }
}

/// A compact set of pending lanes backed by one bit per lane.
///
/// The scheduler keeps a mask of which lanes hold work so lane selection is a
/// single trailing-zeros scan instead of walking every queue.
#[derive(Clone, Copy, PartialEq, Eq, Default, Debug)]
pub struct LaneMask(u8);

impl LaneMask {
    /// The empty mask.
    pub const EMPTY: LaneMask = LaneMask(0);

    /// Creates an empty mask.
    #[must_use]
    pub const fn new() -> LaneMask {
        LaneMask::EMPTY
    }

    /// Adds `lane` to the set.
    pub const fn insert(&mut self, lane: Lane) {
        self.0 |= 1 << lane.index();
    }

    /// Removes `lane` from the set.
    pub const fn remove(&mut self, lane: Lane) {
        self.0 &= !(1 << lane.index());
    }

    /// Whether `lane` is present.
    #[must_use]
    pub const fn contains(self, lane: Lane) -> bool {
        self.0 & (1 << lane.index()) != 0
    }

    /// Whether no lane is present.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// The most urgent lane in the set, or [`None`] when empty.
    #[must_use]
    pub const fn highest(self) -> Option<Lane> {
        if self.0 == 0 {
            return None;
        }
        // Lane 0 (Immediate) is the most urgent and occupies bit 0, so the
        // lowest set bit is the lane to service next.
        Lane::from_index(self.0.trailing_zeros() as usize)
    }

    /// The number of lanes in the set.
    #[must_use]
    pub const fn len(self) -> usize {
        self.0.count_ones() as usize
    }
}
