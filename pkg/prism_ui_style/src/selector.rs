//! Selectors: interaction states, breakpoints and the match context.
//!
//! A class can carry overrides that only apply in certain interaction states
//! (hover, pressed, ...) or at certain responsive breakpoints. [`MatchContext`]
//! captures the current state of the world (which interaction states are active
//! and how wide the viewport is) so the cascade can decide which overrides
//! match.

use core::ops::{BitOr, BitOrAssign};

/// A single interaction state.
///
/// Used both as a map key for per-state overrides on a class and, via
/// [`InteractionStateFlags`], as a member of the set of currently active
/// states. [`InteractionState::Normal`] represents the base state and always
/// matches.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum InteractionState {
    /// The resting state. Always considered active.
    Normal,
    /// The pointer is hovering the element.
    Hover,
    /// The element has keyboard/input focus.
    Focus,
    /// The element is being pressed/activated.
    Pressed,
    /// The element is disabled.
    Disabled,
}

impl InteractionState {
    /// Every non-`Normal` interaction state, in cascade-priority order.
    pub const ALL: [InteractionState; 4] = [
        InteractionState::Hover,
        InteractionState::Focus,
        InteractionState::Pressed,
        InteractionState::Disabled,
    ];

    /// Returns the single-bit mask for this state (`0` for `Normal`).
    #[must_use]
    const fn bit(self) -> u8 {
        match self {
            InteractionState::Normal => 0,
            InteractionState::Hover => 1 << 0,
            InteractionState::Focus => 1 << 1,
            InteractionState::Pressed => 1 << 2,
            InteractionState::Disabled => 1 << 3,
        }
    }
}

/// A hand-rolled bitset of active [`InteractionState`]s (no external deps).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct InteractionStateFlags(u8);

impl InteractionStateFlags {
    /// An empty set (only the implicit `Normal` state is considered active).
    pub const EMPTY: Self = Self(0);

    /// Creates an empty set of states.
    #[must_use]
    pub const fn new() -> Self {
        Self(0)
    }

    /// Returns a copy of this set with `state` added.
    #[must_use]
    pub const fn with(self, state: InteractionState) -> Self {
        Self(self.0 | state.bit())
    }

    /// Adds `state` to this set in place.
    pub fn insert(&mut self, state: InteractionState) {
        self.0 |= state.bit();
    }

    /// Removes `state` from this set in place.
    pub fn remove(&mut self, state: InteractionState) {
        self.0 &= !state.bit();
    }

    /// Returns `true` if `state` is active.
    ///
    /// [`InteractionState::Normal`] is always active.
    #[must_use]
    pub const fn contains(self, state: InteractionState) -> bool {
        match state {
            InteractionState::Normal => true,
            other => self.0 & other.bit() != 0,
        }
    }

    /// Returns `true` if no non-`Normal` state is active.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }
}

impl BitOr for InteractionStateFlags {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

impl BitOr<InteractionState> for InteractionStateFlags {
    type Output = Self;

    fn bitor(self, rhs: InteractionState) -> Self {
        self.with(rhs)
    }
}

impl BitOrAssign<InteractionState> for InteractionStateFlags {
    fn bitor_assign(&mut self, rhs: InteractionState) {
        self.insert(rhs);
    }
}

impl From<InteractionState> for InteractionStateFlags {
    fn from(state: InteractionState) -> Self {
        Self::new().with(state)
    }
}

/// A responsive breakpoint with a minimum-width threshold.
///
/// Breakpoints are mobile-first: a breakpoint matches when the viewport width
/// is greater than or equal to its [`Breakpoint::min_width`]. The thresholds
/// mirror common design-system defaults.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Breakpoint {
    /// The base breakpoint (always matches; `min_width` is `0`).
    Base,
    /// Small (>= 640px).
    Sm,
    /// Medium (>= 768px).
    Md,
    /// Large (>= 1024px).
    Lg,
    /// Extra large (>= 1280px).
    Xl,
}

impl Breakpoint {
    /// Every breakpoint, in ascending min-width order.
    pub const ALL: [Breakpoint; 5] = [
        Breakpoint::Base,
        Breakpoint::Sm,
        Breakpoint::Md,
        Breakpoint::Lg,
        Breakpoint::Xl,
    ];

    /// Returns the minimum viewport width, in logical pixels, at which this
    /// breakpoint becomes active.
    #[must_use]
    pub const fn min_width(self) -> f32 {
        match self {
            Breakpoint::Base => 0.0,
            Breakpoint::Sm => 640.0,
            Breakpoint::Md => 768.0,
            Breakpoint::Lg => 1024.0,
            Breakpoint::Xl => 1280.0,
        }
    }

    /// Returns `true` if `viewport_width` meets this breakpoint's threshold.
    #[must_use]
    pub fn matches(self, viewport_width: f32) -> bool {
        viewport_width >= self.min_width()
    }

    /// Returns the largest breakpoint whose threshold `viewport_width` meets.
    #[must_use]
    pub fn for_width(viewport_width: f32) -> Breakpoint {
        let mut active = Breakpoint::Base;
        for bp in Breakpoint::ALL {
            if bp.matches(viewport_width) {
                active = bp;
            }
        }
        active
    }
}

/// The context against which selectors are matched during a cascade.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MatchContext {
    /// The set of currently active interaction states.
    pub states: InteractionStateFlags,
    /// The current viewport width, in logical pixels.
    pub viewport_width: f32,
}

impl MatchContext {
    /// Creates a context for the given viewport width with no active states.
    #[must_use]
    pub const fn new(viewport_width: f32) -> Self {
        Self {
            states: InteractionStateFlags::new(),
            viewport_width,
        }
    }

    /// Returns a copy of this context with `state` marked active.
    #[must_use]
    pub const fn with_state(mut self, state: InteractionState) -> Self {
        self.states = self.states.with(state);
        self
    }

    /// Returns the largest breakpoint active for this context's viewport width.
    #[must_use]
    pub fn active_breakpoint(&self) -> Breakpoint {
        Breakpoint::for_width(self.viewport_width)
    }
}
