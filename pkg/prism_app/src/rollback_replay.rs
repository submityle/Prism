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
//! it goes. Rolling back **discards the abandoned future**: the snapshots and
//! inputs after the restore point are dropped and the cursor is rebased, so the
//! log never grows a duplicate branch and a short (or aborted) re-simulation
//! cannot leave stale frames behind for a later rollback to resurrect.
//!
//! Use a divergence check to decide *where* to roll back to: compare the
//! per-frame [`FrameHash`](crate::determinism::FrameHash) against a manifest
//! ([`verify_against_manifest`](crate::app::App::verify_against_manifest)) or a
//! peer, take the first divergent frame, and roll back to it.

use alloc::vec::Vec;

use crate::determinism::RecordedInput;
use crate::rollback::SnapshotHistory;
use prism_ecs::component::ComponentId;
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

/// The result of a failed ECS rollback operation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EcsRollbackError {
    /// The app did not install a [`SnapshotHistory`] resource.
    MissingHistory,
    /// A resident component has no registered snapshot clone glue.
    MissingSnapshotComponents(Vec<ComponentId>),
    /// The requested authority revision is outside the retained timeline.
    UnavailableFrame {
        /// The one-based authority revision that was requested.
        frame: u64,
        /// The current local timeline cursor.
        current: u64,
    },
    /// The authority revision is zero; revision zero is the seed, not a step.
    InvalidAuthorityRevision,
}

/// ECS-owned rollback/replay coordination.
///
/// Unlike [`RollbackReplay`], this type does not own a second snapshot ring.
/// The ring is the [`SnapshotHistory`] resource in the ECS [`World`], while this
/// value owns only the small input timeline. That makes the World resource the
/// single source of truth shared by app systems and network integration.
pub struct EcsRollbackReplay<F> {
    inputs: Vec<RecordedInput<F>>,
}

impl<F: Clone> EcsRollbackReplay<F> {
    /// Create an empty coordinator. The caller installs the World history with
    /// [`crate::App::init_snapshot_history`] before committing a frame.
    #[must_use]
    pub const fn new() -> Self {
        Self { inputs: Vec::new() }
    }

    /// The recorded input timeline, indexed by absolute frame number.
    #[must_use]
    pub fn inputs(&self) -> &[RecordedInput<F>] {
        &self.inputs
    }

    /// Commit one predicted frame by capturing the pre-input World state.
    pub fn commit(&mut self, world: &mut World, input: F) -> Result<u64, EcsRollbackError> {
        let mut history = world
            .remove_resource::<SnapshotHistory>()
            .ok_or(EcsRollbackError::MissingHistory)?;
        let frame = match history.try_capture(world) {
            Ok(frame) => frame,
            Err(missing) => {
                world.insert_resource(history);
                return Err(EcsRollbackError::MissingSnapshotComponents(missing));
            }
        };
        world.insert_resource(history);
        self.store_input(frame, input);
        Ok(frame)
    }

    /// Rebase the ECS timeline on an authoritative post-input state and replay
    /// every still-unconfirmed local input.
    ///
    /// `authority_frame` is a one-based authoritative revision: revision `1`
    /// replaces local frame `0`, revision `2` replaces local frame `1`, etc.
    /// The history therefore restores `authority_frame - 1`, applies the
    /// authoritative state there, and replays inputs beginning at
    /// `authority_frame`.
    pub fn rebase_authority(
        &mut self,
        world: &mut World,
        authority_frame: u64,
        mut apply_authority: impl FnMut(&mut World),
        mut simulate: impl FnMut(&mut World, &F),
    ) -> Result<u64, EcsRollbackError> {
        if authority_frame == 0 {
            return Err(EcsRollbackError::InvalidAuthorityRevision);
        }
        let Some(mut history) = world.remove_resource::<SnapshotHistory>() else {
            return Err(EcsRollbackError::MissingHistory);
        };
        let current = history.next_frame();
        let anchor = authority_frame - 1;
        if authority_frame > current || !history.contains(anchor) {
            world.insert_resource(history);
            return Err(EcsRollbackError::UnavailableFrame {
                frame: authority_frame,
                current,
            });
        }
        let Some(replay_slice) = self.inputs.get(authority_frame as usize..current as usize) else {
            world.insert_resource(history);
            return Err(EcsRollbackError::UnavailableFrame {
                frame: authority_frame,
                current,
            });
        };
        let replay_inputs: Vec<F> = replay_slice
            .iter()
            .map(|recorded| recorded.frame.clone())
            .collect();
        let replay_count = replay_inputs.len() as u64;
        if !history.rollback(world, anchor) {
            world.insert_resource(history);
            return Err(EcsRollbackError::UnavailableFrame {
                frame: authority_frame,
                current,
            });
        }
        // The authority state represents the result of the step immediately
        // before the first input we replay.
        apply_authority(world);
        history.truncate_after(anchor);
        history.seek(authority_frame);
        world.insert_resource(history);

        for input in replay_inputs {
            self.commit(world, input.clone())?;
            simulate(world, &input);
        }
        Ok(authority_frame + replay_count)
    }

    fn store_input(&mut self, frame: u64, input: F) {
        let recorded = RecordedInput {
            step: frame,
            frame: input,
        };
        let index = frame as usize;
        if index < self.inputs.len() {
            self.inputs[index] = recorded;
        } else {
            self.inputs.push(recorded);
        }
    }
}

impl<F: Clone> Default for EcsRollbackReplay<F> {
    fn default() -> Self {
        Self::new()
    }
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
        let recorded = RecordedInput {
            step: frame,
            frame: input,
        };
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
        // Abandon the future branch: drop the snapshots and inputs after `frame`
        // and rebase the cursor, so a short (or aborted) re-simulation cannot
        // leave stale frames that a later rollback would resurrect, and
        // `inputs()` never reports frames past the live cursor.
        self.history.truncate_after(frame);
        self.inputs.truncate(frame as usize);
        Some(plan)
    }

    /// Reconcile a divergence at `frame`: roll back and re-simulate the frames
    /// it replaces with the `corrected` inputs.
    ///
    /// `corrected` must hold exactly one input per frame being replaced
    /// (`self.frame() - frame` entries); each is committed and then applied by
    /// `simulate` (which advances the world by one frame). Returns the new
    /// current frame.
    ///
    /// Returns `None` — with **no side effects on `world`** — when `frame` is
    /// not strictly before the current frame, its snapshot has been evicted, or
    /// `corrected.len()` does not match the number of frames replaced.
    pub fn reconcile(
        &mut self,
        world: &mut World,
        frame: u64,
        corrected: &[F],
        mut simulate: impl FnMut(&mut World, &F),
    ) -> Option<u64> {
        let current = self.history.next_frame();
        if frame >= current || corrected.len() as u64 != current - frame {
            return None;
        }
        if !self.history.contains(frame) {
            return None;
        }
        // Restore the pre-frame state (and seek the cursor back to `frame`).
        self.rollback(world, frame)?;
        for input in corrected {
            self.commit(world, input.clone());
            simulate(world, input);
        }
        Some(self.frame())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
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

    /// A rollback discards the abandoned future branch: its snapshots and inputs
    /// are dropped and the cursor is rebased, so a dropped frame can never be
    /// rolled back to again (regression: stale frames used to stay addressable
    /// and a later rollback could resurrect the pre-correction state).
    #[test]
    fn rollback_discards_the_abandoned_future() {
        let (mut w, e) = world(0);
        let mut rr = RollbackReplay::<i32>::new(8);
        for input in 1..=4 {
            forward(&mut rr, &mut w, e, input);
        }
        assert_eq!(rr.frame(), 4);

        let plan = rr.rollback(&mut w, 2).expect("frame 2 is retained");
        assert_eq!(plan, vec![3, 4]);

        assert_eq!(rr.frame(), 2, "the cursor is rebased to the restore point");
        assert_eq!(rr.inputs().len(), 2, "inputs are truncated to the cursor");
        assert!(rr.history().contains(2), "the restore point is kept");
        assert!(!rr.history().contains(3) && !rr.history().contains(4));
        assert!(
            rr.rollback(&mut w, 3).is_none(),
            "a dropped frame can no longer be rolled back to"
        );

        // Re-committing rebuilds the branch from the restore point.
        forward(&mut rr, &mut w, e, 100);
        assert_eq!(rr.frame(), 3);
        assert_eq!(rr.inputs().len(), 3);
        assert!(rr.history().contains(2));
        assert!(!rr.history().contains(4), "frame 4 stays gone");
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

    #[test]
    fn reconcile_replays_with_corrected_inputs() {
        let (mut w, e) = world(0);
        let mut rr = RollbackReplay::<i32>::new(8);
        forward(&mut rr, &mut w, e, 1);
        forward(&mut rr, &mut w, e, 2);
        forward(&mut rr, &mut w, e, 3);
        // Frame 1's late input is 10 (not the recorded 2): reconcile 1..3.
        let frame = rr
            .reconcile(&mut w, 1, &[10, 3], |w, input| sim(w, e, *input))
            .expect("reconcile succeeds");
        assert_eq!(frame, 3);
        assert_eq!(counter(&w, e), 14); // 1 + 10 + 3
        let inputs: Vec<i32> = rr.inputs().iter().map(|r| r.frame).collect();
        assert_eq!(inputs, vec![1, 10, 3]);
    }

    #[test]
    fn reconcile_rejects_a_wrong_length_correction_without_side_effects() {
        let (mut w, e) = world(0);
        let mut rr = RollbackReplay::<i32>::new(8);
        forward(&mut rr, &mut w, e, 1);
        forward(&mut rr, &mut w, e, 2);
        forward(&mut rr, &mut w, e, 3);
        // Two frames need correcting (1 and 2), but only one input is supplied.
        assert!(rr
            .reconcile(&mut w, 1, &[10], |w, input| sim(w, e, *input))
            .is_none());
        assert_eq!(counter(&w, e), 6);
        assert_eq!(rr.frame(), 3);
    }

    #[test]
    fn reconcile_refuses_an_evicted_frame() {
        let (mut w, e) = world(0);
        let mut rr = RollbackReplay::<i32>::new(2);
        forward(&mut rr, &mut w, e, 1);
        forward(&mut rr, &mut w, e, 2);
        forward(&mut rr, &mut w, e, 3);
        // frame 0 was evicted, so its correction cannot be applied.
        assert!(rr
            .reconcile(&mut w, 0, &[9, 9, 9], |w, input| sim(w, e, *input))
            .is_none());
        assert_eq!(counter(&w, e), 6);
    }

    #[test]
    fn ecs_rebase_uses_world_history_and_replays_pending_inputs() {
        let (mut w, e) = world(0);
        w.insert_resource(SnapshotHistory::new(8));
        let mut rr = EcsRollbackReplay::<i32>::new();

        rr.commit(&mut w, 1).expect("frame 0 snapshot");
        sim(&mut w, e, 1);
        rr.commit(&mut w, 2).expect("frame 1 snapshot");
        sim(&mut w, e, 2);
        assert_eq!(counter(&w, e), 3);

        let result = rr
            .rebase_authority(
                &mut w,
                1,
                |world| world.get_mut::<Counter>(e).unwrap().0 = 10,
                |world, input| sim(world, e, *input),
            )
            .expect("authority revision 1 is retained");

        assert_eq!(result, 2);
        assert_eq!(counter(&w, e), 12);
        assert_eq!(w.resource::<SnapshotHistory>().next_frame(), 2);
        assert_eq!(rr.inputs().len(), 2);
    }

    #[test]
    fn ecs_rebase_rejects_missing_history_without_mutating_world() {
        let (mut w, e) = world(7);
        let mut rr = EcsRollbackReplay::<i32>::new();
        let error = rr
            .rebase_authority(&mut w, 1, |_| {}, |world, input| sim(world, e, *input))
            .expect_err("history was not installed");
        assert_eq!(error, EcsRollbackError::MissingHistory);
        assert_eq!(counter(&w, e), 7);
    }
}
