//! Late-input rollback + re-simulation (design §14, §15).
//!
//! [`RollbackReplay`] ties the two rollback primitives together into the loop
//! the design calls for (*"回滚到任意已确认帧并重放输入"*):
//!
//! - [`SnapshotHistory`] bounds *world* state (snapshots get evicted past the
//!   window);
//! - the input log here keeps each frame's input (small, and kept for the whole
//!   session, matching [`InputRecording`](crate::determinism::InputRecording)).
//!
//! The per-frame contract is **commit-then-simulate**: at the start of frame `f`
//! call [`commit`](RollbackReplay::commit) to snapshot the pre-input state and
//! record the frame's input, then advance the simulation with that input. To
//! reconcile a late or corrected input for frame `f`, call
//! [`rollback`](RollbackReplay::rollback): it restores the pre-frame-`f` world
//! and returns the recorded inputs for frames `f..current` to re-simulate (the
//! caller substitutes the corrected frame's input), re-committing each frame as
//! it goes. Re-committing overwrites the frame's snapshot and input in place, so
//! the log never grows a duplicate branch.
//!
//! Use a divergence check to decide *where* to roll back to: compare the
//! per-frame [`FrameHash`](crate::determinism::FrameHash) against a manifest
//! ([`verify_against_manifest`](crate::app::App::verify_against_manifest)) or a
//! peer, take the first divergent frame, and roll back to it.

use crate::determinism::RecordedInput;
use crate::rollback::SnapshotHistory;
use prism_ecs::world::World;

/// A deterministic session that can be rolled back and re-simulated.
///
/// `F` is the application's own per-frame input type (as in
/// [`InputRecording`](crate::determinism::InputRecording)). Frames are numbered
/// `0, 1, 2, ...` in commit order; the snapshot window is bounded by `capacity`.
pub struct RollbackReplay<F> {
    history: SnapshotHistory,
    inputs: Vec<RecordedInput<F>>,
}

impl<F: Clone> RollbackReplay<F> {
    /// A session whose snapshot window retains the most recent `capacity`
    /// frames.
    ///
    /// # Panics
    /// Panics if `capacity` is zero.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            history: SnapshotHistory::new(capacity),
            inputs: Vec::new(),
        }
    }

    /// The bounded snapshot history backing this session.
    #[must_use]
    pub fn history(&self) -> &SnapshotHistory {
        &self.history
    }

    /// The current frame number (frames committed so far).
    #[must_use]
    pub fn frame(&self) -> u64 {
        self.history.next_frame()
    }

    /// The recorded input log, one entry per committed frame in order.
    #[must_use]
    pub fn inputs(&self) -> &[RecordedInput<F>] {
        &self.inputs
    }

    /// Commit the current frame: snapshot `world` (its pre-input state) and
    /// record `input`, then advance the frame number.
    ///
    /// Re-committing a frame (after [`rollback`](Self::rollback) sought back to
    /// it) overwrites that frame's snapshot and input in place.
    ///
    /// # Panics
    /// Panics if a resident component has no snapshot clone glue (same contract
    /// as [`SnapshotHistory::capture`]).
    pub fn commit(&mut self, world: &World, input: F) -> u64 {
        let frame = self.history.next_frame();
        self.history.capture(world);
        let recorded = RecordedInput { step: frame, frame: input };
        let index = frame as usize;
        if index < self.inputs.len() {
            self.inputs[index] = recorded;
        } else {
            self.inputs.push(recorded);
        }
        frame
    }

    /// Roll `world` back to the pre-input state of `frame` and return the
    /// recorded inputs for frames `frame..current` to re-simulate, in order.
    ///
    /// Returns `None` — leaving `world` and the session untouched — when `frame`
    /// is not strictly before the current frame, or when its snapshot has been
    /// evicted (rolled out of the `capacity` window).
    pub fn rollback(&mut self, world: &mut World, frame: u64) -> Option<Vec<F>> {
        let current = self.history.next_frame();
        if frame >= current {
            return None;
        }
        // Slice the plan before touching the world, so a refused rollback has no
        // side effects. `inputs` always spans at least `0..current`.
        let plan: Vec<F> = self.inputs[frame as usize..current as usize]
            .iter()
            .map(|recorded| recorded.frame.clone())
            .collect();
        if !self.history.rollback(world, frame) {
            return None;
        }
        self.history.seek(frame);
        Some(plan)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_ecs::component::Component;
    use prism_ecs::entity::Entity;

    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    struct Counter(i32);
    impl Component for Counter {}

    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    struct Marker;
    impl Component for Marker {}

    fn world(counter: i32) -> (World, Entity) {
        let mut w = World::new();
        w.register_snapshot_component_hashable::<Counter>();
        w.register_snapshot_component::<Marker>();
        let e = w.spawn((Counter(counter), Marker));
        (w, e)
    }

    fn sim(world: &mut World, e: Entity, input: i32) {
        world.get_mut::<Counter>(e).unwrap().0 += input;
    }

    fn counter(world: &World, e: Entity) -> i32 {
        world.get::<Counter>(e).unwrap().0
    }

    /// Commit (snapshot + record) then advance, the per-frame contract.
    fn forward(rr: &mut RollbackReplay<i32>, world: &mut World, e: Entity, input: i32) {
        rr.commit(world, input);
        sim(world, e, input);
    }

    #[test]
    fn commit_records_frames_and_inputs_in_order() {
        let (mut w, e) = world(0);
        let mut rr = RollbackReplay::<i32>::new(8);
        assert_eq!(rr.frame(), 0);
        forward(&mut rr, &mut w, e, 1);
        forward(&mut rr, &mut w, e, 2);
        forward(&mut rr, &mut w, e, 3);
        assert_eq!(rr.frame(), 3);
        assert_eq!(counter(&w, e), 6);
        let steps: Vec<u64> = rr.inputs().iter().map(|r| r.step).collect();
        assert_eq!(steps, vec![0, 1, 2]);
    }

    #[test]
    fn rollback_restores_pre_frame_state_and_returns_replay_plan() {
        let (mut w, e) = world(0);
        let mut rr = RollbackReplay::<i32>::new(8);
        forward(&mut rr, &mut w, e, 1);
        forward(&mut rr, &mut w, e, 2);
        forward(&mut rr, &mut w, e, 3);
        assert_eq!(counter(&w, e), 6);

        // A late input for frame 1 arrives: roll back to its pre-input state.
        let plan = rr.rollback(&mut w, 1).expect("frame 1 is retained");
        assert_eq!(plan, vec![2, 3]);
        assert_eq!(counter(&w, e), 1);
        assert_eq!(rr.frame(), 1);

        // Re-simulate, substituting the corrected input for frame 1.
        forward(&mut rr, &mut w, e, 10);
        forward(&mut rr, &mut w, e, 3);
        assert_eq!(rr.frame(), 3);
        assert_eq!(counter(&w, e), 14); // 1 + 10 + 3
        let inputs: Vec<i32> = rr.inputs().iter().map(|r| r.frame).collect();
        assert_eq!(inputs, vec![1, 10, 3]);
    }

    #[test]
    fn rollback_to_current_or_later_frame_is_refused() {
        let (mut w, e) = world(0);
        let mut rr = RollbackReplay::<i32>::new(4);
        forward(&mut rr, &mut w, e, 5);
        let current = rr.frame();
        assert!(rr.rollback(&mut w, current).is_none());
        assert!(rr.rollback(&mut w, current + 1).is_none());
        // Refused rollbacks leave the world and frame untouched.
        assert_eq!(counter(&w, e), 5);
        assert_eq!(rr.frame(), current);
    }

    #[test]
    fn rollback_to_evicted_frame_is_refused() {
        let (mut w, e) = world(0);
        let mut rr = RollbackReplay::<i32>::new(2);
        forward(&mut rr, &mut w, e, 1);
        forward(&mut rr, &mut w, e, 2);
        forward(&mut rr, &mut w, e, 3);
        // Capacity 2 retains frames 1 and 2; frame 0 has been evicted.
        assert!(rr.rollback(&mut w, 0).is_none());
        assert_eq!(counter(&w, e), 6);
        let plan = rr.rollback(&mut w, 1).expect("frame 1 is retained");
        assert_eq!(plan, vec![2, 3]);
    }
}
