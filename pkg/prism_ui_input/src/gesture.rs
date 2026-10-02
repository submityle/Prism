//! Declarative gesture recognition with a Flutter-style arena.
//!
//! Each [`GestureRecognizer`] consumes pointer events and reports a
//! [`GestureState`]. When several recognizers compete for the same pointer
//! stream, a [`GestureArena`] arbitrates so that exactly one wins:
//!
//! * A recognizer may claim victory eagerly by returning
//!   [`GestureState::Accepted`] (for example a drag once it crosses its slop
//!   threshold). The first such recognizer wins immediately and the rest are
//!   rejected.
//! * A recognizer may bow out by returning [`GestureState::Rejected`] (for
//!   example a tap once the pointer moves too far).
//! * If no recognizer has claimed victory by the time the pointer is released,
//!   the arena *sweeps*: the first still-eligible recognizer wins. This lets a
//!   tap win over a drag that never engaged.
//!
//! The bundled recognizers are [`TapRecognizer`], [`LongPressRecognizer`],
//! [`DragRecognizer`] (with an optional axis constraint and a slop threshold),
//! and [`PinchRecognizer`].

use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use crate::event::{NodeId, PointerEvent, PointerId, PointerKind};
use crate::geometry::{abs_f32, distance_squared, manhattan_span, Point};

/// Current disposition of a gesture recognizer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GestureState {
    /// The gesture might still happen; more input is needed.
    Possible,
    /// The gesture has been recognized and claims the pointer stream.
    Accepted,
    /// The gesture can no longer happen for this pointer stream.
    Rejected,
}

/// A recognizer that turns raw pointer events into a higher-level gesture.
pub trait GestureRecognizer {
    /// Returns the node this recognizer is attached to.
    fn target(&self) -> NodeId;

    /// Feeds one pointer event and returns the updated state.
    fn handle(&mut self, event: &PointerEvent) -> GestureState;

    /// Advances any time-based logic to `now_ms` and returns the updated
    /// state.
    ///
    /// The default implementation ignores time and reports the current state;
    /// recognizers such as [`LongPressRecognizer`] override it.
    fn tick(&mut self, now_ms: u64) -> GestureState {
        let _ = now_ms;
        self.state()
    }

    /// Returns the current state without consuming input.
    fn state(&self) -> GestureState;

    /// Resets the recognizer to its initial, pre-gesture state.
    fn reset(&mut self);
}

/// Recognizes a tap: a press and release without moving beyond `slop`.
pub struct TapRecognizer {
    target: NodeId,
    slop: f32,
    origin: Option<Point<f32>>,
    state: GestureState,
}

impl TapRecognizer {
    /// Creates a tap recognizer for `target` with movement tolerance `slop`.
    pub fn new(target: NodeId, slop: f32) -> Self {
        Self {
            target,
            slop,
            origin: None,
            state: GestureState::Possible,
        }
    }
}

impl GestureRecognizer for TapRecognizer {
    fn target(&self) -> NodeId {
        self.target
    }

    fn handle(&mut self, event: &PointerEvent) -> GestureState {
        if self.state == GestureState::Rejected {
            return self.state;
        }
        match event.kind {
            PointerKind::Down => {
                self.origin = Some(event.position);
                self.state = GestureState::Possible;
            }
            PointerKind::Move => {
                if let Some(origin) = self.origin
                    && distance_squared(origin, event.position) > self.slop * self.slop
                {
                    self.state = GestureState::Rejected;
                }
            }
            PointerKind::Up => {
                self.state = match self.origin {
                    Some(origin)
                        if distance_squared(origin, event.position) <= self.slop * self.slop =>
                    {
                        GestureState::Accepted
                    }
                    _ => GestureState::Rejected,
                };
            }
            PointerKind::Cancel => self.state = GestureState::Rejected,
            PointerKind::Enter | PointerKind::Leave => {}
        }
        self.state
    }

    fn state(&self) -> GestureState {
        self.state
    }

    fn reset(&mut self) {
        self.origin = None;
        self.state = GestureState::Possible;
    }
}

/// Recognizes a long press: holding still for at least `hold_ms`.
pub struct LongPressRecognizer {
    target: NodeId,
    slop: f32,
    hold_ms: u64,
    origin: Option<Point<f32>>,
    down_at: Option<u64>,
    state: GestureState,
}

impl LongPressRecognizer {
    /// Creates a long-press recognizer requiring `hold_ms` of still contact.
    pub fn new(target: NodeId, slop: f32, hold_ms: u64) -> Self {
        Self {
            target,
            slop,
            hold_ms,
            origin: None,
            down_at: None,
            state: GestureState::Possible,
        }
    }
}

impl GestureRecognizer for LongPressRecognizer {
    fn target(&self) -> NodeId {
        self.target
    }

    fn handle(&mut self, event: &PointerEvent) -> GestureState {
        if self.state == GestureState::Rejected {
            return self.state;
        }
        match event.kind {
            PointerKind::Down => {
                self.origin = Some(event.position);
                self.down_at = Some(event.timestamp_ms);
                self.state = GestureState::Possible;
            }
            PointerKind::Move => {
                if let Some(origin) = self.origin
                    && distance_squared(origin, event.position) > self.slop * self.slop
                {
                    self.state = GestureState::Rejected;
                }
                if self.state == GestureState::Possible {
                    self.state = self.evaluate(event.timestamp_ms);
                }
            }
            PointerKind::Up => {
                // Released before the hold elapsed (otherwise a prior tick or
                // move would already have accepted).
                if self.state == GestureState::Possible {
                    self.state = self.evaluate(event.timestamp_ms);
                    if self.state == GestureState::Possible {
                        self.state = GestureState::Rejected;
                    }
                }
            }
            PointerKind::Cancel => self.state = GestureState::Rejected,
            PointerKind::Enter | PointerKind::Leave => {}
        }
        self.state
    }

    fn tick(&mut self, now_ms: u64) -> GestureState {
        if self.state == GestureState::Possible {
            self.state = self.evaluate(now_ms);
        }
        self.state
    }

    fn state(&self) -> GestureState {
        self.state
    }

    fn reset(&mut self) {
        self.origin = None;
        self.down_at = None;
        self.state = GestureState::Possible;
    }
}

impl LongPressRecognizer {
    /// Returns `Accepted` when enough time has elapsed, else `Possible`.
    fn evaluate(&self, now_ms: u64) -> GestureState {
        match self.down_at {
            Some(start) if now_ms.saturating_sub(start) >= self.hold_ms => GestureState::Accepted,
            _ => GestureState::Possible,
        }
    }
}

/// Axis a [`DragRecognizer`] is allowed to engage along.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum DragAxis {
    /// Engage on movement in any direction.
    #[default]
    Both,
    /// Engage only when horizontal movement dominates.
    Horizontal,
    /// Engage only when vertical movement dominates.
    Vertical,
}

/// Recognizes a drag once the pointer moves past `slop` along the allowed axis.
pub struct DragRecognizer {
    target: NodeId,
    slop: f32,
    axis: DragAxis,
    origin: Option<Point<f32>>,
    translation: Point<f32>,
    state: GestureState,
}

impl DragRecognizer {
    /// Creates a drag recognizer with movement tolerance `slop` and the given
    /// axis constraint.
    pub fn new(target: NodeId, slop: f32, axis: DragAxis) -> Self {
        Self {
            target,
            slop,
            axis,
            origin: None,
            translation: Point::new(0.0, 0.0),
            state: GestureState::Possible,
        }
    }

    /// Returns the accumulated translation since the drag started.
    pub fn translation(&self) -> Point<f32> {
        self.translation
    }

    /// Returns `true` when `dx`/`dy` satisfies the axis constraint.
    fn axis_allows(&self, dx: f32, dy: f32) -> bool {
        match self.axis {
            DragAxis::Both => true,
            DragAxis::Horizontal => abs_f32(dx) >= abs_f32(dy),
            DragAxis::Vertical => abs_f32(dy) >= abs_f32(dx),
        }
    }
}

impl GestureRecognizer for DragRecognizer {
    fn target(&self) -> NodeId {
        self.target
    }

    fn handle(&mut self, event: &PointerEvent) -> GestureState {
        if self.state == GestureState::Rejected {
            return self.state;
        }
        match event.kind {
            PointerKind::Down => {
                self.origin = Some(event.position);
                self.translation = Point::new(0.0, 0.0);
                self.state = GestureState::Possible;
            }
            PointerKind::Move => {
                if let Some(origin) = self.origin {
                    let dx = event.position.x - origin.x;
                    let dy = event.position.y - origin.y;
                    if self.state == GestureState::Accepted {
                        self.translation = Point::new(dx, dy);
                    } else if dx * dx + dy * dy >= self.slop * self.slop {
                        if self.axis_allows(dx, dy) {
                            self.translation = Point::new(dx, dy);
                            self.state = GestureState::Accepted;
                        } else {
                            self.state = GestureState::Rejected;
                        }
                    }
                }
            }
            PointerKind::Up => {
                if self.state != GestureState::Accepted {
                    self.state = GestureState::Rejected;
                }
            }
            PointerKind::Cancel => self.state = GestureState::Rejected,
            PointerKind::Enter | PointerKind::Leave => {}
        }
        self.state
    }

    fn state(&self) -> GestureState {
        self.state
    }

    fn reset(&mut self) {
        self.origin = None;
        self.translation = Point::new(0.0, 0.0);
        self.state = GestureState::Possible;
    }
}

/// Recognizes a two-pointer pinch, reporting a scale factor.
///
/// # Scale metric
///
/// Scale is defined as the ratio of the current span to the span measured when
/// the second pointer arrived. To avoid a disallowed square root, the span is
/// the Manhattan (L1) distance between the two contacts rather than the
/// Euclidean distance. This is monotonic with zoom — a value above `1.0` means
/// the contacts spread apart, below `1.0` means they came together — which is
/// what pinch consumers need, at the cost of not being a true geometric ratio.
pub struct PinchRecognizer {
    target: NodeId,
    threshold: f32,
    pointers: BTreeMap<PointerId, Point<f32>>,
    start_span: Option<f32>,
    current_span: f32,
    state: GestureState,
}

impl PinchRecognizer {
    /// Creates a pinch recognizer that engages once the span changes by at
    /// least `threshold` layout units.
    pub fn new(target: NodeId, threshold: f32) -> Self {
        Self {
            target,
            threshold,
            pointers: BTreeMap::new(),
            start_span: None,
            current_span: 0.0,
            state: GestureState::Possible,
        }
    }

    /// Returns the current scale factor relative to the engagement span.
    ///
    /// Returns `1.0` before two pointers are tracked or when the starting span
    /// is degenerate.
    pub fn scale(&self) -> f32 {
        match self.start_span {
            Some(start) if start > 0.0 => self.current_span / start,
            _ => 1.0,
        }
    }

    /// Returns the number of pointers currently tracked.
    pub fn active_pointers(&self) -> usize {
        self.pointers.len()
    }

    /// Recomputes the span across the two lowest-id pointers.
    fn recompute_span(&mut self) {
        let mut iter = self.pointers.values();
        match (iter.next(), iter.next()) {
            (Some(&a), Some(&b)) => self.current_span = manhattan_span(a, b),
            _ => self.current_span = 0.0,
        }
    }
}

impl GestureRecognizer for PinchRecognizer {
    fn target(&self) -> NodeId {
        self.target
    }

    fn handle(&mut self, event: &PointerEvent) -> GestureState {
        if self.state == GestureState::Rejected {
            return self.state;
        }
        match event.kind {
            PointerKind::Down => {
                self.pointers.insert(event.pointer, event.position);
                if self.pointers.len() >= 2 {
                    self.recompute_span();
                    self.start_span = Some(self.current_span);
                }
            }
            PointerKind::Move => {
                if let Some(slot) = self.pointers.get_mut(&event.pointer) {
                    *slot = event.position;
                    self.recompute_span();
                    if let Some(start) = self.start_span
                        && abs_f32(self.current_span - start) >= self.threshold
                    {
                        self.state = GestureState::Accepted;
                    }
                }
            }
            PointerKind::Up | PointerKind::Cancel => {
                self.pointers.remove(&event.pointer);
                if self.pointers.len() < 2 && self.state == GestureState::Possible {
                    // Not enough contacts remain to form a pinch.
                    self.start_span = None;
                }
                self.recompute_span();
            }
            PointerKind::Enter | PointerKind::Leave => {}
        }
        self.state
    }

    fn state(&self) -> GestureState {
        self.state
    }

    fn reset(&mut self) {
        self.pointers.clear();
        self.start_span = None;
        self.current_span = 0.0;
        self.state = GestureState::Possible;
    }
}

/// One competitor in a [`GestureArena`].
struct ArenaMember {
    recognizer: Box<dyn GestureRecognizer>,
    rejected: bool,
}

/// Arbitrates between competing recognizers for a single pointer stream.
#[derive(Default)]
pub struct GestureArena {
    members: Vec<ArenaMember>,
    winner: Option<usize>,
}

impl GestureArena {
    /// Creates an empty arena.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds `recognizer` to the arena, returning its member index.
    pub fn add(&mut self, recognizer: Box<dyn GestureRecognizer>) -> usize {
        let index = self.members.len();
        self.members.push(ArenaMember {
            recognizer,
            rejected: false,
        });
        index
    }

    /// Returns the winning member index, if one has been decided.
    pub fn winner(&self) -> Option<usize> {
        self.winner
    }

    /// Returns the number of members.
    pub fn len(&self) -> usize {
        self.members.len()
    }

    /// Returns `true` when the arena has no members.
    pub fn is_empty(&self) -> bool {
        self.members.is_empty()
    }

    /// Returns the state of the member at `index`, if it exists.
    pub fn member_state(&self, index: usize) -> Option<GestureState> {
        self.members.get(index).map(|m| m.recognizer.state())
    }

    /// Feeds a pointer event to every eligible member and returns the winner,
    /// if one is decided by this event.
    ///
    /// A pointer release triggers a sweep when no member has claimed victory.
    pub fn route(&mut self, event: &PointerEvent) -> Option<usize> {
        if let Some(w) = self.winner {
            // Still forward to the winner so it can update (e.g. drag delta).
            self.members[w].recognizer.handle(event);
            return Some(w);
        }
        for index in 0..self.members.len() {
            if self.members[index].rejected {
                continue;
            }
            let state = self.members[index].recognizer.handle(event);
            match state {
                GestureState::Rejected => self.members[index].rejected = true,
                GestureState::Accepted => {
                    self.declare_winner(index);
                    return Some(index);
                }
                GestureState::Possible => {}
            }
        }
        if event.kind == PointerKind::Up {
            return self.sweep();
        }
        None
    }

    /// Advances time-based members to `now_ms` and returns the winner, if one
    /// is decided by this tick.
    pub fn tick(&mut self, now_ms: u64) -> Option<usize> {
        if let Some(w) = self.winner {
            return Some(w);
        }
        for index in 0..self.members.len() {
            if self.members[index].rejected {
                continue;
            }
            let state = self.members[index].recognizer.tick(now_ms);
            match state {
                GestureState::Rejected => self.members[index].rejected = true,
                GestureState::Accepted => {
                    self.declare_winner(index);
                    return Some(index);
                }
                GestureState::Possible => {}
            }
        }
        None
    }

    /// Awards victory to the first still-eligible member.
    pub fn sweep(&mut self) -> Option<usize> {
        if self.winner.is_some() {
            return self.winner;
        }
        for index in 0..self.members.len() {
            if !self.members[index].rejected {
                self.declare_winner(index);
                return Some(index);
            }
        }
        None
    }

    /// Marks `index` the winner and rejects every other member.
    fn declare_winner(&mut self, index: usize) {
        self.winner = Some(index);
        for (other, member) in self.members.iter_mut().enumerate() {
            if other != index {
                member.rejected = true;
                member.recognizer.reset();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::PointerId;

    fn down(x: f32, y: f32, t: u64) -> PointerEvent {
        PointerEvent::new(PointerId::new(1), PointerKind::Down, Point::new(x, y), t)
    }
    fn mv(x: f32, y: f32, t: u64) -> PointerEvent {
        PointerEvent::new(PointerId::new(1), PointerKind::Move, Point::new(x, y), t)
    }
    fn up(x: f32, y: f32, t: u64) -> PointerEvent {
        PointerEvent::new(PointerId::new(1), PointerKind::Up, Point::new(x, y), t)
    }

    #[test]
    fn tap_accepts_on_release_without_movement() {
        let mut tap = TapRecognizer::new(NodeId::new(1), 4.0);
        assert_eq!(tap.handle(&down(0.0, 0.0, 0)), GestureState::Possible);
        assert_eq!(tap.handle(&up(1.0, 1.0, 10)), GestureState::Accepted);
    }

    #[test]
    fn tap_rejects_when_moved_past_slop() {
        let mut tap = TapRecognizer::new(NodeId::new(1), 4.0);
        tap.handle(&down(0.0, 0.0, 0));
        assert_eq!(tap.handle(&mv(20.0, 0.0, 5)), GestureState::Rejected);
        assert_eq!(tap.handle(&up(20.0, 0.0, 10)), GestureState::Rejected);
    }

    #[test]
    fn drag_accepts_after_slop_and_tracks_translation() {
        let mut drag = DragRecognizer::new(NodeId::new(1), 5.0, DragAxis::Both);
        drag.handle(&down(0.0, 0.0, 0));
        assert_eq!(drag.handle(&mv(2.0, 2.0, 5)), GestureState::Possible);
        assert_eq!(drag.handle(&mv(20.0, 0.0, 10)), GestureState::Accepted);
        assert_eq!(drag.translation(), Point::new(20.0, 0.0));
    }

    #[test]
    fn horizontal_drag_rejects_vertical_motion() {
        let mut drag = DragRecognizer::new(NodeId::new(1), 5.0, DragAxis::Horizontal);
        drag.handle(&down(0.0, 0.0, 0));
        assert_eq!(drag.handle(&mv(0.0, 30.0, 5)), GestureState::Rejected);
    }

    #[test]
    fn long_press_accepts_after_hold() {
        let mut lp = LongPressRecognizer::new(NodeId::new(1), 4.0, 500);
        lp.handle(&down(0.0, 0.0, 0));
        assert_eq!(lp.tick(200), GestureState::Possible);
        assert_eq!(lp.tick(500), GestureState::Accepted);
    }

    #[test]
    fn long_press_rejects_on_early_release() {
        let mut lp = LongPressRecognizer::new(NodeId::new(1), 4.0, 500);
        lp.handle(&down(0.0, 0.0, 0));
        assert_eq!(lp.handle(&up(0.0, 0.0, 100)), GestureState::Rejected);
    }

    #[test]
    fn pinch_accepts_when_span_grows() {
        let mut pinch = PinchRecognizer::new(NodeId::new(1), 10.0);
        let a_down = PointerEvent::new(
            PointerId::new(1),
            PointerKind::Down,
            Point::new(0.0, 0.0),
            0,
        );
        let b_down = PointerEvent::new(
            PointerId::new(2),
            PointerKind::Down,
            Point::new(10.0, 0.0),
            0,
        );
        pinch.handle(&a_down);
        assert_eq!(pinch.handle(&b_down), GestureState::Possible);
        assert_eq!(pinch.active_pointers(), 2);
        let b_move = PointerEvent::new(
            PointerId::new(2),
            PointerKind::Move,
            Point::new(40.0, 0.0),
            5,
        );
        assert_eq!(pinch.handle(&b_move), GestureState::Accepted);
        assert!(pinch.scale() > 1.0);
    }

    #[test]
    fn arena_drag_beats_tap_eagerly() {
        let mut arena = GestureArena::new();
        let tap = arena.add(Box::new(TapRecognizer::new(NodeId::new(1), 5.0)));
        let drag = arena.add(Box::new(DragRecognizer::new(
            NodeId::new(1),
            5.0,
            DragAxis::Both,
        )));
        arena.route(&down(0.0, 0.0, 0));
        assert_eq!(arena.winner(), None);
        let winner = arena.route(&mv(30.0, 0.0, 5));
        assert_eq!(winner, Some(drag));
        assert_eq!(arena.member_state(tap), Some(GestureState::Possible));
    }

    #[test]
    fn arena_sweep_lets_tap_win_on_release() {
        let mut arena = GestureArena::new();
        let tap = arena.add(Box::new(TapRecognizer::new(NodeId::new(1), 5.0)));
        let _drag = arena.add(Box::new(DragRecognizer::new(
            NodeId::new(1),
            5.0,
            DragAxis::Both,
        )));
        arena.route(&down(0.0, 0.0, 0));
        let winner = arena.route(&up(1.0, 1.0, 10));
        assert_eq!(winner, Some(tap));
    }

    #[test]
    fn arena_reports_len_and_empty() {
        let mut arena = GestureArena::new();
        assert!(arena.is_empty());
        arena.add(Box::new(TapRecognizer::new(NodeId::new(1), 5.0)));
        assert_eq!(arena.len(), 1);
        assert!(!arena.is_empty());
    }
}
