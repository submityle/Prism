//! Multi-touch input: raw touch events and the per-frame touch tracker.
//!
//! A touchscreen reports independent contact points, each identified by a
//! platform-assigned id that is stable for the lifetime of a touch (from
//! [`TouchPhase::Started`] to [`TouchPhase::Ended`] or
//! [`TouchPhase::Canceled`]). Backends emit [`TouchInput`] events; [`Touches`]
//! folds them into queryable per-point state with begin/end edges, mirroring
//! the `ButtonInput` edge model but keyed by touch id and carrying position.

use alloc::collections::{BTreeMap, BTreeSet, btree_map};

/// The lifecycle stage a [`TouchInput`] event reports for one contact point.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum TouchPhase {
    /// The contact point was placed on the surface.
    Started,
    /// The contact point moved while remaining on the surface.
    Moved,
    /// The contact point was lifted off the surface normally.
    Ended,
    /// The contact point was invalidated by the system (gesture, interruption).
    Canceled,
}

impl TouchPhase {
    /// Whether this phase ends the touch (either [`Ended`](Self::Ended) or
    /// [`Canceled`](Self::Canceled)).
    #[must_use]
    pub const fn is_terminal(self) -> bool {
        matches!(self, Self::Ended | Self::Canceled)
    }
}

/// A single raw touch event for one contact point.
///
/// Kept [`Copy`] so it can live inside the `Copy` `InputEvent` stream. Positions
/// are in logical pixels in the surface's coordinate space (origin top-left).
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct TouchInput {
    /// Platform id, stable for the lifetime of this touch.
    pub id: u64,
    /// What happened to the contact point.
    pub phase: TouchPhase,
    /// Horizontal position in logical pixels.
    pub x: f32,
    /// Vertical position in logical pixels.
    pub y: f32,
}

impl TouchInput {
    /// Builds a touch event.
    #[must_use]
    pub const fn new(id: u64, phase: TouchPhase, x: f32, y: f32) -> Self {
        Self { id, phase, x, y }
    }
}

/// The tracked state of a single active contact point, accumulated by
/// [`Touches`] across the frames the touch is live.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Touch {
    /// Platform id of this contact point.
    pub id: u64,
    /// The most recent reported phase.
    pub phase: TouchPhase,
    /// Current position (logical pixels).
    pub position: (f32, f32),
    /// Position when the touch started (logical pixels).
    pub start_position: (f32, f32),
    /// Position at the previous update (logical pixels).
    pub previous_position: (f32, f32),
}

impl Touch {
    /// Position delta since the previous update.
    #[must_use]
    pub fn delta(&self) -> (f32, f32) {
        (
            self.position.0 - self.previous_position.0,
            self.position.1 - self.previous_position.1,
        )
    }

    /// Position delta since the touch started.
    #[must_use]
    pub fn distance_from_start(&self) -> (f32, f32) {
        (
            self.position.0 - self.start_position.0,
            self.position.1 - self.start_position.1,
        )
    }
}

/// Per-frame multi-touch tracker.
///
/// Feed every [`TouchInput`] via [`process`](Self::process) as events arrive,
/// then call [`clear`](Self::clear) once per frame to retire the
/// `just_pressed` / `just_released` edges and drop points whose touch ended.
/// Active points iterate in ascending id order for deterministic replay.
#[derive(Clone, PartialEq, Debug, Default)]
pub struct Touches {
    active: BTreeMap<u64, Touch>,
    just_pressed: BTreeSet<u64>,
    just_released: BTreeSet<u64>,
}

impl Touches {
    /// Creates an empty tracker.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Applies one raw event, updating the active point and its edges.
    pub fn process(&mut self, event: TouchInput) {
        let pos = (event.x, event.y);
        match event.phase {
            TouchPhase::Started => {
                self.just_pressed.insert(event.id);
                self.active.insert(
                    event.id,
                    Touch {
                        id: event.id,
                        phase: TouchPhase::Started,
                        position: pos,
                        start_position: pos,
                        previous_position: pos,
                    },
                );
            }
            TouchPhase::Moved => {
                if let Some(touch) = self.active.get_mut(&event.id) {
                    touch.previous_position = touch.position;
                    touch.position = pos;
                    touch.phase = TouchPhase::Moved;
                }
            }
            TouchPhase::Ended | TouchPhase::Canceled => {
                self.just_released.insert(event.id);
                if let Some(touch) = self.active.get_mut(&event.id) {
                    touch.previous_position = touch.position;
                    touch.position = pos;
                    touch.phase = event.phase;
                }
            }
        }
    }

    /// Retires per-frame edges: clears the just-pressed/just-released sets and
    /// removes points whose touch has ended, while advancing live points'
    /// previous position to the current one. Call once per frame.
    pub fn clear(&mut self) {
        self.just_pressed.clear();
        self.just_released.clear();
        self.active.retain(|_, touch| !touch.phase.is_terminal());
        for touch in self.active.values_mut() {
            touch.previous_position = touch.position;
        }
    }

    /// Removes all tracked touches and edges.
    pub fn reset(&mut self) {
        self.active.clear();
        self.just_pressed.clear();
        self.just_released.clear();
    }

    /// Number of currently active contact points.
    #[must_use]
    pub fn len(&self) -> usize {
        self.active.len()
    }

    /// Whether there are no active contact points.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.active.is_empty()
    }

    /// Looks up a single active touch by id.
    #[must_use]
    pub fn get(&self, id: u64) -> Option<&Touch> {
        self.active.get(&id)
    }

    /// Whether the touch with `id` began this frame.
    #[must_use]
    pub fn just_pressed(&self, id: u64) -> bool {
        self.just_pressed.contains(&id)
    }

    /// Whether the touch with `id` ended this frame.
    #[must_use]
    pub fn just_released(&self, id: u64) -> bool {
        self.just_released.contains(&id)
    }

    /// Iterates active touches in ascending id order.
    pub fn iter(&self) -> btree_map::Values<'_, u64, Touch> {
        self.active.values()
    }
}
